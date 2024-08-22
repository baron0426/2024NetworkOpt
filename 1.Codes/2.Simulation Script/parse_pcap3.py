from scapy.all import *
import csv
import os


def get_ip_lookup_dict(csv_file):
    ip_lookup_dict = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 2:
                raise Exception("CSV Format Error: IP Lookup Table CSV must have 2 columns.")
            ip_lookup_dict[row[1]] = row[0]
    return ip_lookup_dict


def parse_pcap(node_num, pcap_files, ip_lookup_dict, output_file):
    with open(output_file, 'w') as file:
        file.write('day,time,src_node,dst_node\n')
        for t in range(len(pcap_files)):
            for pcap_file in pcap_files[t]:
                packets = rdpcap(pcap_file)
                for packet in packets:
                    src_node = int(ip_lookup_dict[packet[IP].src])
                    dst_node = int(ip_lookup_dict[packet[IP].dst])
                    if (src_node is not node_num) and (dst_node is not node_num):
                        continue  # disregard forwarding packets for each node.
                    file.write("{},{},{},{}\n".format(t, packet.time, src_node, dst_node))


def generate_file_list(path, nNodes, nTimes):
    file_list = [[[] for _ in range(nTimes)] for _ in range(nNodes)]
    for filename in os.listdir(path):
        num_part = filename.split('_')[-1]
        time = int(num_part.split('-')[0])
        node_num = int(num_part.split('-')[1])
        file_list[node_num][time].append(path + filename)
    return file_list


def sort_result(csv_file):
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)
    sorted_rows = sorted(rows, key=lambda x: (
    float(x[0]), int(x[2]), int(x[3]), float(x[1])))  # Assuming Time column is numeric
    # Write the sorted rows back to the original file, including the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        writer.writerows(sorted_rows)


if __name__ == '__main__':
    base_dir = '../dataset/'
    output_base_dir = '../results/'
    test_name = 'simulation_final'
    test_types = ['no_priority', 'Spectral', 'CrENMF33', 'GrENMF5']
    nTimes = 116
    nNodes = 22
    for test_type in test_types:
        ip_lookup_dict = get_ip_lookup_dict('{}/{}/IpLookupTable_{}.csv'.format(base_dir, test_name, test_type))
        pcap_folder = '{}/{}/{}/'.format(base_dir, test_name, test_type)
        output_folder = '{}/{}/{}/'.format(output_base_dir, test_name, test_type)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_list = generate_file_list(pcap_folder, nNodes, nTimes)
        for node_num in range(nNodes):
            output_file = output_folder + str(node_num) + '.csv'
            parse_pcap(node_num, file_list[node_num], ip_lookup_dict, output_file)
        for filename in os.listdir(output_folder):
            sort_result(output_folder + filename)
