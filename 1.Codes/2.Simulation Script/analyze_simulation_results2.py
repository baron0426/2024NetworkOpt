import csv
import numpy as np


def filter_traffic(nodes, nDays, file):
    (src, dst) = nodes
    src = int(src)
    dst = int(dst)
    ret_val = [[] for _ in range(nDays)]

    with open(file, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if int(row[2]) is not src:
                continue
            if int(row[3]) is not dst:
                continue
            ret_val[int(row[0])].append(row[1])
    return ret_val


def get_traffic_metrics(nodes, nDays, files_path):
    sender_timestamps = filter_traffic(nodes, nDays, '{}/{}.csv'.format(files_path, nodes[0]))
    receiver_timestamps = filter_traffic(nodes, nDays, '{}/{}.csv'.format(files_path, nodes[1]))
    retVal = [[] for _ in range(nDays)]
    for d in range(nDays):
        if (len(sender_timestamps[d]) == 0) or (len(receiver_timestamps[d]) == 0):
            retVal[d] = [d, nodes[0], nodes[1], len(sender_timestamps[d]), len(receiver_timestamps[d]), 0, 0]
        else:
            s_times = np.array(sender_timestamps[d]).astype('float32')
            r_times = np.array(receiver_timestamps[d]).astype('float32')
            average_eted = np.average(r_times - s_times)
            total_delay = r_times[-1] - s_times[0]
            # print("Pkt Sent: {}; Pkt Received:{}; Average ETED:{}s; Total Delay:{}s".format(len(sender_timestamps), len(receiver_timestamps), average_eted, total_delay))
            retVal[d] = [d, nodes[0], nodes[1], len(s_times), len(r_times), average_eted,
                         total_delay]
    return retVal


if __name__ == '__main__':
    base_dir = '../results/'
    test_name = 'simulation_final'
    test_types = ['no_priority', 'Spectral', 'CrENMF33', 'GrENMF5']
    nNodes = 22
    nDays = 116

    for test_type in test_types:
        test_file_path = '{}/{}/{}/'.format(base_dir, test_name, test_type)
        output_file = '{}/{}/report_{}.csv'.format(base_dir, test_name, test_type)
        with open(output_file, mode='w', newline='') as report:
            writer = csv.writer(report)
            writer.writerow(['day', 'src_node', 'dst_node', 'pkt_sent', 'pkt_received', 'average_eted', 'total_delay'])
            for k in range(nNodes):
                for m in range(nNodes):
                    if k == m:
                        continue  # no loopback traffic!
                    nodes = (k, m)
                    res = get_traffic_metrics(nodes, nDays, test_file_path)
                    # print(res)
                    writer.writerows(res)
