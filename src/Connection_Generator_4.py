import time

#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

#先生成连接矩阵在转换成simulater需要的连接列表
np.random.seed(2022)#固定随机种子
nb_pn = 840
nb_kc = 4000
nb_en = 1


# 复现适配：不调用 kpsewhich，不依赖系统安装 TeX 字体。
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['svg.fonttype'] = 'path'


def s_generator(pre_pop_size, post_pop_size, group=1, sparseness=5, bias=0, w_u=1.4, w_std=0, block=True):
    # pre_pop_size, post_pop_size: pre- and post-synaptic neuron numbers
    # bias: standard deviation of sparseness
    # group: number of neuron subgroups [1,15)
    # sparseness : how many pre neuron per post neuron connects to [5,8]
    # w_u = weight mean
    # w_std = weight standard deviation 标准差
    # block = True : pre & post neurons are divided into isolated subgroups [i+m]-[j+n], [x+m]-[y+n]
    # block = False: neurons connects to nearby post- neurons i-[j, j+n]

    S = np.full(shape=(pre_pop_size, post_pop_size), dtype=float, fill_value= np.nan)
    for i in range(post_pop_size):
        nb_pre_p_post = int(sparseness + bias*np.random.rand())#nb=number,抽取多少次来源编号
        #生成 nb_pre_p_post 个随机权重，保存在 sw_pre_p_post 中
        sw_pre_p_post = np.random.normal(loc=w_u, scale=w_std, size=nb_pre_p_post).tolist()


        #确定来源编号的上界和下界
        pre_group_size = pre_pop_size // group
        if block is True:
            pre_group_low = i * group // post_pop_size * pre_group_size
        #   pre_group_high = np.clip(pre_group_low + pre_group_size, 0, pre_pop_size)
        else:
            pre_group_low = int(i * group / post_pop_size * pre_group_size)
        pre_group_high = pre_group_low + pre_group_size#上界

        #随机抽取size个整数
        s_pre_p_post_i = np.random.randint(low=pre_group_low, high=pre_group_high, size=nb_pre_p_post,dtype=int)

        s_pre_p_post_i[s_pre_p_post_i >=(pre_pop_size)] -= pre_pop_size#没用？
        #将生成的权重写入矩阵
        S[s_pre_p_post_i, i] = sw_pre_p_post
        # print( nb_pre_p_post, pre_group_low,pre_group_high)


    return S


def scatter_hist(S):#画图代码，没仔细看
    idx = np.where(S >= 0)
    x = idx[0]
    y = idx[1]
    w = S[idx].flatten()

    fig = plt.figure(figsize=(11, 11),dpi=100)
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.15]
    rect_histy = [left + width + spacing, bottom, 0.15, height]

    # start with a square Figure


    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax, )
    ax_histy = fig.add_axes(rect_histy, sharey=ax, )




    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    # fig, ax = plt.subplots()
    # ax.imshow(data)
    # ax = sns.heatmap(data)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    cmap = matplotlib.cm.summer
    sc = ax.scatter(x=x, y=y, c=(w), marker=',', s=1, cmap = cmap, norm=norm)
    ax.set_xlabel('pre- neuron index', size= 30)
    ax.set_ylabel('post- neuron index',size= 30)
    # ax.set_xlim([-50,max(x)+50],auto=False)
    # ax.set_ylim([-50,max(y)+50],auto=False)
    # now determine nice limits by hand:
    binwidth = 0.25
    xmax = np.max(np.abs(x))
    ymax = np.max(np.abs(y))
    # xmax = np.max(x)
    # ymax = np.max(y)
    xlim = (int(xmax / binwidth) + 1) * binwidth
    ylim = (int(ymax / binwidth) + 1) * binwidth

    xbins = np.arange(0, xlim + binwidth, binwidth)
    ybins = np.arange(0, ylim + binwidth, binwidth)
    ax_histx.hist(x, bins=xbins, histtype = 'step')
    ax_histx.set_title('number of connections per pre neuron',rotation='horizontal')
    ax_histy.hist(y, bins=ybins, histtype = 'step', orientation='horizontal')
    ax_histy.set_title('number of connections per post neuron', rotation='vertical',x=0.1,y=0.1)


    cbar = plt.colorbar(sc)

    # tick_locator = ticker.MaxNLocator(nbins=3)
    # cbar.locator = tick_locator
    cbar.set_ticks(np.linspace(0,1,3))
    cbar.set_label("weights", loc='center')

    cbar.set_ticklabels( ('0', '0.5', '1'))
    plt.show()
   # plt.savefig(str(time.ctime())+'.svg')




#把S转成连接列表，(x,y,z)=(来源编号，目标编号，大于0的权重)，PN神经元
def generate_pnlist(S):
    lista= []
    idx = np.where(S >0)
    x = idx[0]
    y = idx[1]
    w = S[idx].flatten()
    for i in range(len(x)):
        lista.append((x[i],y[i],w[i]))
    return lista

#只保留连接位置 ，KC神经元
def generate_kclist(S):
    lista= []
    idx = np.where(S > 0)
    x = idx[0]
    y = idx[1]
    for i in range(len(x)):
        lista.append((x[i],y[i]))
    return lista
#840*4000连接矩阵
S_pn2kc = s_generator(pre_pop_size=nb_pn,post_pop_size=nb_kc,group=1,sparseness=5, bias = 1, w_u = 0.3,w_std=0.15, block=True)
#4000*4000连接矩阵
S_kc2kc = s_generator(pre_pop_size=nb_kc,post_pop_size=nb_kc,group=1,sparseness=1000,bias = 0, w_u=0.5,block=True)


# S_pn2kc = s_generator(pre_pop_size=nb_pn,post_pop_size=nb_kc,group=1,sparseness=4,bias=1,w_u= 0.5, w_std=0, block=True)
#
# S_kc2kc = s_generator(pre_pop_size=nb_kc,post_pop_size=nb_kc,group=4,sparseness=500,w_u=0.2,block=True)

#fig = plt.figure(figsize=(12, 10),dpi=100)

#
# scatter_hist(S_pn2kc)
#
# scatter_hist(S_kc2kc)


S_pn2kc = generate_pnlist(S_pn2kc)
    
#S_pn2kc = generate_kclist(S_pn2kc)
S_kc2kc = generate_kclist(S_kc2kc)

#
print ('PN2KC list:', len(S_pn2kc))
print ('KC2KC list:', len(S_kc2kc))
#
