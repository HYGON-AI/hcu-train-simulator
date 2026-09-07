# Collect measured communication bandwidth

## get 2 nodes bw date

'''
cd tools/bandwidth
bash run_two_nodes.sh
python collect_bandwidth.py --input-dir logs/bw1000_2 --output ../../src/hcu_train_simulator/communication/bandwidth/bw1000_2.txt
'''

## get 1 nodes bw date

'''
cd tools/bandwidth
bash run_one_node.sh
python collect_bandwidth.py --input-dir logs/bw1000_1 --output ../../src/hcu_train_simulator/communication/bandwidth/bw1000_1.txt
'''

## result in bw1000_1.txt and bw1000_2.txt
