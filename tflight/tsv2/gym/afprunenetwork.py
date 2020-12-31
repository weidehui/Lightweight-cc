from stable_baselines.common.policies import FeedForwardPolicy
from tsv2.common.simple_arg_parse import arg_or_default
arch=[32,16]
pruning_perc=25
arch[0]=arch[0]-8
arch[1]=arch[1]-4

class StuMlpPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(StuMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess

