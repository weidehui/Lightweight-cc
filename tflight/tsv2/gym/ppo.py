import pprint

import gym
import tianshou as ts
import torch
import numpy as np
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
import network_sim00
import network_sim01
import network_sim02
import network_sim03
import network_sim04
import network_sim05
import network_sim06
import network_sim07
import network_sim08
import network_sim10
import network_sim11
import network_sim12
import network_sim13
import network_sim14
import network_sim15
import network_sim16
import network_sim17
import network_sim18
import network_sim200
import network_sim201
import network_sim202
import network_sim203
import network_sim204
import network_sim205
import network_sim206
import network_sim207
import network_sim208
import network_sim210
import network_sim211
import network_sim212
import network_sim213
import network_sim214
import network_sim215
import network_sim216
import network_sim217
import network_sim218
from tianshou.policy import PPOPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-py', action='store_true', default=False)
    parser.add_argument('--debug-env', action='store_true', default=False)
    parser.add_argument('--enable-load-train', action='store_true', default=False)
    parser.add_argument('--enable-load-test', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=500)  # 10
    parser.add_argument('--step-per-epoch', type=int, default=8192)  # 1000
    parser.add_argument('--collect-per-step', type=int, default=10)  # 10
    parser.add_argument('--repeat-per-collect', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--episode-per-test', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.8)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(model_path='model/src-200',args=get_args(),cwnd_use=0,env_num=0,space_is_two=1):
    if(space_is_two):
        env_name='PccNs-v'+'2'+str(cwnd_use)+str(env_num)
    else:
        env_name='PccNs-v'+str(cwnd_use)+str(env_num)
    env= gym.make(env_name)
    args.max_action = env.action_space.high[0]

    #seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape

    env = DummyVectorEnv(
        [lambda: gym.make(env_name) for _ in range(args.training_num)])
    env.seed(args.seed)

    net = Net(args.layer_num, args.state_shape, device=args.device,hidden_layer_size=32)
    actor = ActorProb(
        net,args.action_shape,
        args.max_action,args.device,
        hidden_layer_size=32
    ).to(args.device)
    critic = Critic(net, device=args.device,hidden_layer_size=32).to(args.device)

    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)


    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = ts.policy.PPOPolicy(
        actor,critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm, # clipping gradients in back propagation, defaults to None
        eps_clip=args.eps_clip, #ϵ  in LCLIP in the original paper, defaults to 0.2
        vf_coef=args.vf_coef, #weight for value loss, defaults to 0.5
        ent_coef=args.ent_coef, #weight for value loss, defaults to 0.5
        reward_normalization=args.rew_norm,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda
    )

    train_collector = ts.data.Collector(policy, env, ts.data.ReplayBuffer(size=args.buffer_size))
    test_collector = ts.data.Collector(policy, env)

    # log
    log_path = os.path.join(model_path+"/log")
    writer = SummaryWriter(log_path)

    policy.load_state_dict(torch.load(model_path+'/policy.pth'))

    # Let's watch its performance!
    env = gym.make(env_name)
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    test_ppo()
