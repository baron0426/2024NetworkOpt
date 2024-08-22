import csv
import numpy as np


def filter_traffic(nodes, file):
    (src, dst) = nodes
    src = int(src)
    dst = int(dst)
    ret_val = []
    with open(file, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if int(row[1]) is not src:
                continue
            if int(row[2]) is not dst:
                continue
            ret_val.append(row[0])
    return ret_val


def get_traffic_metrics(nodes, files_path):
    sender_timestamps = filter_traffic(nodes, '{}/{}.csv'.format(files_path, nodes[0]))
    receiver_timestamps = filter_traffic(nodes, '{}/{}.csv'.format(files_path, nodes[1]))
    sender_timestamps = np.array(sender_timestamps).astype('float32')
    receiver_timestamps = np.array(receiver_timestamps).astype('float32')
    if (len(sender_timestamps) == 0) or (len(receiver_timestamps) == 0):
        return [nodes[0], nodes[1], len(sender_timestamps), len(receiver_timestamps), 0, 0]
    average_eted = np.average(receiver_timestamps - sender_timestamps)
    total_delay = receiver_timestamps[-1] - sender_timestamps[0]
    # print("Pkt Sent: {}; Pkt Received:{}; Average ETED:{}s; Total Delay:{}s".format(len(sender_timestamps), len(receiver_timestamps), average_eted, total_delay))
    return [nodes[0], nodes[1], len(sender_timestamps), len(receiver_timestamps), average_eted, total_delay]


if __name__ == '__main__':
    base_dir = '../results/'
    test_name = 'simulation_test'
    test_types = ['priority', 'no_priority']
    nNodes = 5

    for test_type in test_types:
        test_file_path = '{}/{}/{}/'.format(base_dir, test_name, test_type)
        output_file = '{}/{}/report_{}.csv'.format(base_dir, test_name, test_type)
        with open(output_file, mode='w', newline='') as report:
            writer = csv.writer(report)
            writer.writerow(['src_node', 'dst_node', 'pkt_sent', 'pkt_received', 'average_eted', 'total_delay'])
            for k in range(nNodes):
                for m in range(nNodes):
                    if k == m:
                        continue  # no loopback traffic!
                    nodes = (k, m)
                    res = get_traffic_metrics(nodes, test_file_path)
                    writer.writerow(res)
