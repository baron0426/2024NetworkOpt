#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace ns3;
using namespace std;

void SendPacket(Ptr<Node> sender, InetSocketAddress dest, double interarrivalMean, double trafficEndTime) {
    static double pktSize = 1000;  // size of a single packet in Bytes

    Ptr<Socket> socket = Socket::CreateSocket(sender, UdpSocketFactory::GetTypeId());
    Ptr<Packet> packet = Create<Packet>(pktSize);
    socket->SendTo(packet, 0, dest);
    if(Simulator::Now().GetSeconds() > trafficEndTime) return;
    // Generate random number following exponential distribution
    Ptr<ExponentialRandomVariable> intervalRV = CreateObject<ExponentialRandomVariable> ();
    intervalRV->SetAttribute ("Mean", DoubleValue (interarrivalMean));
    intervalRV->SetAttribute ("Bound", DoubleValue (100.0*interarrivalMean));
    // Time nextInterval = Seconds(intervalRV->GetValue());
    Simulator::Schedule(Seconds(intervalRV->GetValue()), &SendPacket, sender, dest, interarrivalMean, trafficEndTime);
}

void OutputIpLookupTable(const vector<vector<Ipv4Address>>& data, const string& filename){
    ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    for(size_t i = 0; i < data.size(); i++){
        for(size_t j = 0; j<data[i].size(); j++){
            file << i << ",";
            data[i][j].Print(file);
            file << endl;
        }
    }
    file.close();
}

void readCSV(const string &filename, vector<vector<int>> &matrix, const int& N, const int& M) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    string line;
    for (int i = 0; i < N; ++i) {
        if (!getline(file, line)) {
            cerr << "Error reading line " << i << " from file: " << filename << endl;
            return;
        }
        stringstream ss(line);
        string value;
        for (int j = 0; j < M; ++j) {
            if (!getline(ss, value, ',')) {
                cerr << "Error reading value " << j << " from line " << i << " in file: " << filename << endl;
                return;
            }
            matrix[i][j] = stoi(value);
        }
    }
    file.close();
}

int main (int argc, char *argv[]){
    if(argc < 2){
        cout << "Error!" << endl;
        return -1;
    }

    size_t n = 22;  // number of nodes
    size_t t = atoi(argv[1]);
    uint16_t appPort = 9; // port number of the UDP sockets
    double trafficStartTime = 1.0; // in seconds
    double trafficDuration = 1.0;   // in seconds
    // double waitDuration = trafficDuration*500;
    double simulationEndTime = trafficStartTime + 100;


    // Adjacency Matrix (Unit: Kbps)
    vector<vector<int>> adjacencyMatrix(n, vector<int>(n, 0));
    string adjacencyMatrixFilePath = "./topology_matrix/topology_matrix.csv";
    readCSV(adjacencyMatrixFilePath, adjacencyMatrix, n, n);

    // Traffic Lambda Matrix (Unit: pkt/s)
    vector<vector<int>> trafficProbabilityMatrix(n, vector<int>(n,0));
    string trafficMatrixBaseFilePath = "./traffic_matrices/";
    readCSV(trafficMatrixBaseFilePath+to_string(t)+".csv", trafficProbabilityMatrix, n, n);

    // Create Nodes
    NodeContainer nodes;
    nodes.Create(n);

    // Install internet stack
    InternetStackHelper stack;
    stack.Install (nodes);

    // Create Links
    PointToPointHelper p2pHelper;
    p2pHelper.SetChannelAttribute ("Delay", StringValue ("0ms"));
    p2pHelper.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1p"));

    // Create IP lookup table for nodes
    int linkNum = 0;
    vector<vector<Ipv4Address>> ipLookupTable(n, vector<Ipv4Address>(0));

    for(size_t i = 0; i < n; i++){
        for(size_t j = i+1; j < n; j++){
            if(adjacencyMatrix[i][j] <= 0) continue;
            string linkRate = to_string(adjacencyMatrix[i][j])+"Kbps";
            p2pHelper.SetDeviceAttribute ("DataRate", StringValue (linkRate.c_str()));
            NetDeviceContainer devices = p2pHelper.Install (nodes.Get (i), nodes.Get (j));
            // Install traffic control
            TrafficControlHelper tch;
            tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc");
            tch.Install (devices);
            // Assign IP address
            Ipv4AddressHelper address;
            string ipBase = "10.0."+ to_string(linkNum) +".0";
            address.SetBase (ipBase.c_str(), "255.255.255.0");
            Ipv4InterfaceContainer interfaces = address.Assign (devices);
            ipLookupTable[i].push_back(interfaces.GetAddress(0));
            ipLookupTable[j].push_back(interfaces.GetAddress(1));
            linkNum++;
        }
    }

    OutputIpLookupTable(ipLookupTable, "IpLookupTable_no_priority.csv");

    // Set up sink application in each node
    for (size_t i = 0; i < n; i++) {
        //Sink App Installation
        PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", InetSocketAddress("0.0.0.0", appPort));
        ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(i));
        sinkApps.Start(Seconds(0.0));
    }

    // Setup Routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

    // Generate Traffic

    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < n; j++){
            if(i == j) continue;
            if(trafficProbabilityMatrix[i][j] == 0) continue; // no traffic to send!
            // trafficProbabilityMatrix[i][j] === [Sender: Node i, Receiver: Node j] @ Time t
            Ptr<Node> sender = nodes.Get(i);
            InetSocketAddress dest = InetSocketAddress(ipLookupTable[j][0], appPort);
            dest.SetTos(4); // TODO: change the priority! 4=high priority, 12 = low priority
            double interarrivalMean = 10.0/trafficProbabilityMatrix[i][j];
            double start = trafficStartTime;
            double end = start + trafficDuration;
            Simulator::Schedule(Seconds(start), &SendPacket, sender, dest, interarrivalMean, end);
        }
    }

    // Enable PCAP tracing on both nodes
    p2pHelper.EnablePcapAll("real2_no_priority_"+to_string(t));

    // Run simulation
    Simulator::Stop(Seconds(simulationEndTime));
    Simulator::Run ();
    Simulator::Destroy ();
    return 0;
}
