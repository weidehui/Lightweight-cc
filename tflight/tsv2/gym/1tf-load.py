import gym
import network_sim0
import network_sim1
from loaded import LoadedModelAgent
from tqdm import tqdm
import os,shutil
import plotall
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path) 
def move_files(src_path,dst_path):
    files = os.listdir(src_path)
    for file in files:
        if  os.path.exists(os.path.join(dst_path,file)): 
            os.remove(os.path.join(dst_path,file))
        shutil.move(os.path.join(src_path,file),dst_path)


filename=[['jiandan-false-2-old'],[]]

for cwnd_use in range(0,2):
    env = gym.make('PccNs-v'+str(cwnd_use))
    for f in filename[cwnd_use]:
        a=LoadedModelAgent('model-tf/'+f)
        print("PATH: ",'model-tf/'+f)
        a.network_prune()

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step1(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step2(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step3(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step4(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step5(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step6(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step7(act)

        obs=env.reset()
        for i in tqdm(range(1600)):
            act=a.act(obs)
            obs,reward,done,info=env.step8(act)

        mkdir('result-tf/'+f)
        move_files('data','result-tf/'+f)
        plotall.plotRes([f],'-tf')