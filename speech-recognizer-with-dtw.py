import numpy as np
import time
from os import listdir, getcwd
from os.path import isfile, join

DATA_PATH = join(getcwd(), 'ProjectData')
TRAIN_DATA_PATH = join(DATA_PATH, 'TrainData')
TEST_DATA_PATH = join(DATA_PATH, 'EvalData')

def get_commands(command_path):
    '''Gets possible commands list'''
    return open(command_path).readlines()

def get_train_set(path):
    '''Gets train instances with ids, path and command label'''
    data_instances = open(join(path, 'instances.txt'), 'w', encoding='Cp1254')
    commands_dirs_by_readers = [join(path,f) for f in listdir(path) if not isfile(join(path, f))]
    instances = [[f[:-4], join(join(directory, 'komutlar'), f),  open(join(join(directory, 'komutlar'), f[:-3] + 'txt'),  encoding="Cp1254").read()] for directory in commands_dirs_by_readers for f in listdir(join(directory, 'komutlar')) if f.endswith('.mfc')]
    data_instances.write('\n'.join([','.join(instance) for instance in instances]))
    data_instances.close()    
    return instances    

def get_test_set(path):
    '''Gets paths of test instances'''
    return [[f, join(path,f)]for f in listdir(path) if f.endswith('.mfc')]

def get_mfc_file(path):
    '''Loads mfc file as matrix'''
    return np.loadtxt(path)

def local_distance(v1, v2, ord=1):
    '''Gets local distance between vectors with given order'''
    return np.linalg.norm(v1-v2, ord=ord)

def dtw_distance(template, test):
    col = template.shape[0]
    row = test.shape[0]
    distance = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if i == 0 and j == 0:
                distance[i,j] = local_distance(template[j], test[i])
            elif j == 0:
                distance[i,j] = distance[i-1,j] + local_distance(template[j], test[i])
            elif i == 0:
                distance[i,j] = distance[i,j-1] + local_distance(template[j], test[i])
            else:
                distance[i,j] = min(distance[i-1,j-1], distance[i,j-1], distance[i-1,j]) + local_distance(template[j], test[i])
    return distance[row-1,col-1]

def find_nearest_neighbor(test, templates):
    minD = dtw_distance(get_mfc_file(templates[0][1]), test)
    minI = 0
    command = templates[0][2] 
    for i in range(1, len(templates)):
        D = dtw_distance(get_mfc_file(templates[i][1]), test)
        if D < minD:
            minD = D
            minI = i
            command = templates[i][2]
    return minD, minI, command

def predict_commands(test_instances, train_instances):
    predictions = open(join(TEST_DATA_PATH, 'predictions_l1.txt'), 'w', encoding='Cp1254')
    start = time.time()
    for test_instance in test_instances:
        test = get_mfc_file(test_instance[1])
        distance, index, command = find_nearest_neighbor(test, train_instances)
        print(test_instance[0], command)
        predictions.write(test_instance[0] + ' ' + command + '\n')
    predictions.close()
    end = time.time()
    print("time passed", end - start)

def main():
	train_instances = get_train_set(TRAIN_DATA_PATH)
	test_instances = get_test_set(TEST_DATA_PATH)    
	predict_commands(test_instances, train_instances)


if __name__ == '__main__':
	main()                        