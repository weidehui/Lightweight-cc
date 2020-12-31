import ppo
import os,shutil
from tqdm import tqdm
import plotall
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
def move_files(src_path,dst_path):
    files = os.listdir(src_path)
    for file in files:
        if  os.path.exists(os.path.join(dst_path,file)): 
            os.remove(os.path.join(dst_path,file))
        shutil.move(os.path.join(src_path,file),dst_path)
#models=[
#        [['old-false-1e12','new-false-1e12'],['old-false-2','new-false-2','src-200']],
#        [['old-true-1e12','new-true-1e12'],['old-true-2','new-true-2']]       
#        ]
models=[
        [[],['32-fuza-model-2-50-false-old-small-v2']],
        [[],[]]
        ]
for space_is_two in range(0,2):
    for cwnd_use in range(0,2):
        for model in models[cwnd_use][space_is_two]:
            model_path='model/'+model
            print('load:'+model)
            for i in tqdm(range(0,9),desc = model):
                ppo.test_ppo(env_num=i,model_path=model_path,cwnd_use=cwnd_use,space_is_two=space_is_two)
            mkdir('result0/'+model)
            move_files('data','result0/'+model)
            plotall.plotRes([model],'0')