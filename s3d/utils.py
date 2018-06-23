import scipy as sp
import pandas as pd
import networkx as nx
import palettable
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, r2_score

sns.set_context("paper", font_scale=2)

def obtain_metric(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1_binary = f1_score(y_true, y_pred, average='binary')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    r2 = r2_score(y_true, y_pred)

    auc_macro = roc_auc_score(y_true, y_score)
    auc_micro = roc_auc_score(y_true, y_score, 'micro')

    d = {'accuracy': acc, 'auc_macro': auc_macro,
         'auc_micro': auc_micro, 'f1_binary':f1_binary,
         'f1_macro': f1_macro, 'f1_micro': f1_micro, 'r2': r2}
    return pd.Series(d)

def visualize_cv(performance_file,
                 split_version,
                 validation_metric = 'auc_micro',
                 color_list=None, aspect=2.7, size=2.5):
    ''' visualize the increment of r-squared of s3d model, for invidual data splits

        Parameters
        ----------
        performance_file : str
            the performance of validation sets based on cross val
        validation_metric : str
            choose a metric to plot. by default uses auc (micro average)
        color_list : list
            a list of colors to be used for each step;
            length must be the same as the number of steps (aka the number of selected features)
        aspect, size : float
            aspect and size the the parameters in sns.factorplot() function
            see: https://seaborn.pydata.org/generated/seaborn.factorplot.html
    '''

    df = pd.read_csv(performance_file)
    df = df.query("split_version==@split_version").drop(columns='split_version')
    cv_df = df.groupby(['lambda_', 'num_features']).mean()[[validation_metric, 'train_r2']]
    cv_df = cv_df.reset_index().melt(id_vars=['lambda_', 'num_features'],
                                     value_vars=[validation_metric, 'train_r2'])
    cv_df['lambda_'] = ['%.5f'%l for l in cv_df.lambda_.values]

    ## color list
    if color_list is None:
        color_list = eval('palettable.colorbrewer.qualitative.Pastel1_'+str(cv_df.lambda_.unique().size)+'.mpl_colors')
    elif len(color_list) < cv_df.lambda_.unique().size:
        raise ValueError('color_list does not have enough colors ({}) for lambdas ({})'\
                          .format(len(color_list), cv_df.lambda_.unique().size))

    ## circle the highest one
    best_series = cv_df.query("variable==@validation_metric")\
                          .sort_values('value',
                                       ascending=False).iloc[0]
    best_x, best_y, best_lambda_ = best_series.loc[['num_features',
                                                    'value',
                                                    'lambda_']].values.astype(float)


    fp = sns.factorplot(x='num_features', y='value', data=cv_df, fit_reg=False,
                        palette=color_list,
                        aspect=aspect, size=size, row='variable', row_order=['train_r2', validation_metric],
                        hue='lambda_', kind='point', sharey=False, legend_out=True)
    fp.axes[0,0].set_title('Training Performance', fontdict={'size': 14})
    fp.axes[1,0].set_title('Validation Performance', fontdict={'size': 14})
    fp.axes[0,0].set_ylabel('$R^2$\non training', fontdict={'size': 12})
    fp.axes[1,0].set_ylabel('{}\non heldout'.format(validation_metric), fontdict={'size': 12})

    fp.axes[1,0].plot([best_x-1], [best_y], 'o', mfc='none', ms=20, mew=2, color='k')
    fp.axes[1,0].axhline(best_y, xmax=(best_x)/10, color='k', lw=2, ls='--')
    fp.axes[1,0].axvline(best_x-1, ymax=0.9, color='k', lw=2, ls='--')

    fp._legend.set_title('$\lambda$ values', prop={'size':14})
    fp.fig.subplots_adjust(top=0.9,right=0.8)

    fp.set_xlabels('Features Selected', fontdict={'size': 12})
    return fp, best_x, best_y, best_lambda_, split_version

def visualize_s3d_steps(model_folder, figsize=(8,7), color_list=None):
    ''' visualize the increment of r-squared of s3d model
        Parameters
        ----------
        model_folder : str
            the folder of a given trained model
        figsize : tuple/list
            a 2-element tuple or list that controls the size of the figure
        color_list : list
            a list of colors to be used for each step;
            length must be the same as the number of steps (aka the number of selected features)
    '''
    if not model_folder.endswith('/'):
        model_folder += '/'
    df = pd.read_csv(model_folder+'R2improvements.csv')
    #print(df)
    df = df.T.sort_values(0).T
    fig, ax = plt.subplots(figsize=figsize)
    left_base = 0
    width = 0.05
    y = pd.np.arange(df.shape[1])
    if color_list is None:
        color_list = eval('palettable.colorbrewer.qualitative.Pastel1_'+str(df.shape[0])+'.mpl_colors')
    elif len(color_list) < df.shape[0]:
        raise ValueError('color_list does not have enough colors ({}) for lambdas ({})'\
                          .format(len(color_list), df.shape[0]))
    ## highlight the x labels as well
    for i in range(df.shape[0]):
        ax.axvline(x=left_base, linestyle='--', color='k')
        #df.loc[i].plot(kind='barh', color=color_list[i], ax=ax, left=left_base, label='Step %d'%(i+1))
        i_bar = ax.barh(y, df.loc[i].values, color=color_list[i], left=left_base, label='Step %d'%(i+1))
        ## highlihght the highest bar
        #print(df.loc[i].reset_index(drop=True).idxmax())
        i_max = df.loc[i].reset_index(drop=True).idxmax()
        #print(i_bar.patch)
        i_bar.patches[i_max].set_linewidth(3)
        i_bar.patches[i_max].set_edgecolor('k')
        left_base += df.loc[i].max()

    ax.set_xlabel(r'$R^2$')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_yticks(y)
    ax.set_yticklabels(df.columns)
    _ = ax.legend(loc='upper center', fancybox=True, shadow=True,
                  ncol=df.shape[0], bbox_to_anchor=(0.5, 1.1))
    return (fig, ax)

def visualize_s3d_model_reader(model_folder, dim, thres):
    levels = pd.read_csv(model_folder+'/levels.csv')
    chosen_features = levels.loc[:,'best_feature'].values

    # `splits.csv`: bins for each feature
    splits = []
    with open(model_folder+'splits.csv') as f:
        for line in f:
            ## [1:] to skip feature names
            splits.append([float(x) for x in line.split()[0].split(',')[1:]])
    for split in splits:
        if split[0] == split[1]: # sinlge point interval
            if split[1] == 0:
                split[1] = 0.1
            else:
                split[0] = split[0]-1

    # `ybar_tree.csv`: the average y values in each bin
    ybars = []
    with open(model_folder+'ybar_tree.csv') as f:
        for line in f:
            ybars.append([float(x) for x in line.split()[0].split(',')])

    #`N_tree.csv`: the number of data points in each bin
    Ns = []
    with open(model_folder+'/N_tree.csv') as f:
        for line in f:
            Ns.append([int(x) for x in line.split()[0].split(',')])
    splits_at_dim = splits[:dim]
    Ns_sublist = Ns[dim]
    intensity = ybars[dim]

    intensity_mesh = pd.np.reshape(pd.np.array(intensity),
                                   list(map(lambda x:len(x)-1, splits_at_dim)))
    Ns_mesh = pd.np.reshape(pd.np.array(Ns_sublist),
                            list(map(lambda x:len(x)-1, splits_at_dim)))
    pred_masked = ((intensity_mesh >= thres) & (Ns_mesh >0)).astype(int)
    intensity_masked = pd.np.ma.masked_where((Ns_mesh==0)|(intensity_mesh<=0),
                                             intensity_mesh)
    return splits_at_dim, Ns_mesh, intensity_masked, pred_masked, chosen_features[:dim]

def visualize_s3d_model(dim, splits_at_dim, Ns_mesh,
                        intensity_masked, pred_masked, chosen_features,
                        vmin, vmax, xscale, yscale,
                        xlab_x, xlab_y, ylab_x, ylab_y,
                        unit_w, unit_h, hspace, wspace,
                        fontsize, cbar_aspect, labelpad,
                        intensity_cmap, pred_cmap, scale):
    nrows = ncols = 1
    if dim == 3:
        ncols = len(splits_at_dim[0])-1
    if dim == 4:
        nrows = len(splits_at_dim[0])-1
        ncols = len(splits_at_dim[1])-1
    #print(nrows, ncols) 
    ## 1 row; 2 columns
    if dim==3:
        outer_grid = gridspec.GridSpec(2, 1, hspace=hspace)
        figsize = (ncols*scale*unit_h, nrows*unit_w*scale)
        fig = plt.figure(figsize=figsize)
    else:
        outer_grid = gridspec.GridSpec(1, 2, wspace=wspace)
        figsize = (ncols*scale*unit_w, nrows*unit_h*scale)
        fig = plt.figure(figsize=figsize)
    masked_list = [intensity_masked, pred_masked]
    cmap_list = [intensity_cmap, pred_cmap]
    ## inner grid: for empirical probablity (ybar)
    inner_grid_ybar = gridspec.GridSpecFromSubplotSpec(ncols=ncols, nrows=nrows, subplot_spec=outer_grid[0])
    ## inner grid: for binary prediction (dichitomize ybar given a threshold)
    inner_grid_pred = gridspec.GridSpecFromSubplotSpec(ncols=ncols, nrows=nrows, subplot_spec=outer_grid[1])
    for idx, inner_g in enumerate([inner_grid_ybar, inner_grid_pred]):
        masked_arr = masked_list[idx]
        ax_l = list()
        for i in range(nrows):
            for j in range(ncols):
                ax = plt.Subplot(fig, inner_g[i, j])
                if dim == 2:
                    mesh_map = masked_arr
                elif dim == 3:
                    mesh_map = masked_arr[j,:,:]
                else:
                    mesh_map = masked_arr[nrows-i-1,j,:]
                if vmin is None:
                    vmin=masked_arr.min()
                if vmax is None:
                    vmax=masked_arr.max()
                im = ax.pcolormesh(splits_at_dim[dim-2], splits_at_dim[dim-1],
                                   mesh_map.T, cmap=cmap_list[idx],
                                   vmin=vmin, vmax=vmax
                                  )
                ax.set_xlim(max(1, min(splits_at_dim[dim-2])), max(splits_at_dim[dim-2]))
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.set_ylim(max(1, min(splits_at_dim[dim-1])), max(splits_at_dim[dim-1]))

                if ((dim==3 and idx==1) or (dim==4)) and (i==nrows-1):
                    xlab_str = '{}'.format(chosen_features[dim-2])
                    xlab_str += '\n$[{}, {})$'.format(splits_at_dim[dim-3][j],
                                                       splits_at_dim[dim-3][j+1])
                    ax.set_xlabel(xlab_str, size=fontsize*.7)
                elif dim < 4 and i == nrows-1:
                    ax.set_xlabel(chosen_features[0], size=fontsize*.7)
                else:
                    ax.set_xticks([])

                if j == 0:
                    if dim == 4 and idx==0:
                        ylab_str = '$[{}, {})$\n'.format(splits_at_dim[0][nrows-i-1],
                                                          splits_at_dim[0][nrows-i])
                        ylab_str += '{}'.format(chosen_features[dim-1])
                    elif (dim==2 and idx==0) or (dim==3):
                        ylab_str = '{}'.format(chosen_features[dim-1])
                    else:
                        ylab_str=''
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                    #print(dim, idx, ylab_str)
                else:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    ylab_str=''
                ax.set_ylabel(ylab_str, size=fontsize*.8)
                fig.add_subplot(ax, sharey=True, sharex=True)
                ax_l.append(ax)
        # colorbar
        cb = fig.colorbar(im,  ax=ax_l,
                          fraction=0.046, pad=0.04,
                          aspect=cbar_aspect)
        if idx == 0:
            cb.set_label('$E[Y]$',rotation=270, size=fontsize*0.8, labelpad=labelpad)
        elif idx==1:
            cb.set_ticks([0, 1])
            cb.set_label('Prediction',rotation=270, size=fontsize*0.8, labelpad=labelpad//2)
        if idx == 0 and dim==4:
            if ylab_x is None:
                ylab_x = 0.05*scale
            if ylab_y is None:
                ylab_y = 0.6/scale
            fig.text(x=ylab_x, y=ylab_y, s=chosen_features[0],
                     size=fontsize, rotation=90)
    #outer_grid.tight_layout(fig)
    if dim == 3:
        if xlab_x is None:
            xlab_x = .33*scale
        if xlab_y is None:
            xlab_y = -0.4/scale
        fig.text(x=xlab_x, y=xlab_y, s=chosen_features[dim-3], size=fontsize)
        #fig.suptitle(chosen_features[dim-3], size=15, y=-0.25, x=0.45)
    elif dim == 4:
        if xlab_x is None:
            xlab_x = 0.45
        if xlab_y is None:
            xlab_y = 0.01
        fig.text(x=xlab_x, y=xlab_y, s=chosen_features[dim-3], size=fontsize)
    else:
        outer_grid.update(wspace=0.7)
    return fig

def visualize_s3d(model_folder, dim, thres=0.5, chosen_features=None,
                  vmin=None, vmax=None, xscale='log', yscale='log',
                  xlab_x=None, xlab_y=None, ylab_x=None, ylab_y=None,
                  unit_w=7, unit_h=2.5, hspace=0.7, wspace=0.5,
                  fontsize=20, cbar_aspect=20, labelpad=20,
                  intensity_cmap='Greens',
                  pred_cmap=ListedColormap(['#fbfafa', 'green']), scale=1
                 ):
    splits_at_dim, Ns_mesh, intensity_masked, pred_masked, features = visualize_s3d_model_reader(model_folder, dim, thres)
    if chosen_features is None:
        chosen_features = features
    fig = visualize_s3d_model(dim, splits_at_dim, Ns_mesh,
                              intensity_masked, pred_masked,
                              chosen_features, vmin, vmax,
                              xscale, yscale,
                              xlab_x, xlab_y, ylab_x, ylab_y,
                              unit_w, unit_h, hspace, wspace,
                              fontsize, cbar_aspect, labelpad,
                              intensity_cmap, pred_cmap, scale)
    return fig


def visualize_feature_network_contruct(model_folder, node_label_mapping, color_choice, isolated_option):
    ''' helper function to visualize feature network '''

    r2_improv_df = pd.read_csv(model_folder+'R2improvements.csv')
    ## construct graph
    g = nx.DiGraph()
    selected_nodes = r2_improv_df.idxmax(axis=1).values
    unselected_nodes = set(r2_improv_df.columns) - set(selected_nodes)
    isolated_nodes = list(nx.isolates(g))
    g.add_nodes_from(selected_nodes, color=color_choice['selected'])
    g.add_nodes_from(unselected_nodes, color=color_choice['unselected'])
    for idx, series in r2_improv_df.iterrows():
        if idx+1 == r2_improv_df.shape[0]:
            break
        ## selected one has the largest val
        selected_feature = series.idxmax()
        #print(selected_feature)
        ## for each unselected ones, create edges
        for feature in r2_improv_df.columns:
            if feature == selected_feature:
                continue
            ## skip edges w/o r2 improvement
            w = series.loc[feature] - r2_improv_df.loc[idx+1, feature]
            if w == 0:
                continue
            g.add_edge(feature, selected_feature, weight=w)
    draw_node_list = set(g)
    if isolated_option == 'remove':
        draw_node_list = set(g) - set(nx.isolates(g))
    else:
        if isolated_option != 'color':
            warnings.warn('isolated option not understood (remove/color).\nUse color option by default')
        color_dict = nx.get_node_attributes(g, 'color')
        for u in nx.isolates(g):
            color_dict[u] = color_choice['isolated']
        nx.set_node_attributes(g, color_dict, 'color')

    if node_label_mapping is not None:
        nx.relabel_nodes(g, node_label_mapping, False)

    return g, draw_node_list

def visualize_feature_network(model_folder,
                              node_label_mapping=None,
                              color_choice=dict(zip(['selected', 'unselected', 'isolated'],
                                                    palettable.colorbrewer.qualitative.Pastel1_3.hex_colors)),
                              isolated_option='remove', layout=None,
                              w_scale=200, node_size=800, figsize=(10,10),
                              arrowsize=20, arrowstyle='->',
                              edge_color='k', edge_weight_list=None,
                              edge_kwargs={}, node_kwargs={}, label_kwargs={},
                             ):
    g, draw_node_llist = visualize_feature_network_contruct(model_folder, node_label_mapping,
                                                            color_choice,
                                                            isolated_option)

    if edge_weight_list is None:
        edge_weight_list = list(nx.get_edge_attributes(g, 'weight').values())
        edge_weight_list = abs(pd.np.array(edge_weight_list)*w_scale)

    node_color_list = list(nx.get_node_attributes(g, 'color').values())

    if layout is None:
        layout = nx.circular_layout(g)

    fig, ax = plt.subplots(figsize=figsize)
    labels = nx.draw_networkx_labels(g, layout, ax=ax, **label_kwargs)
    edges = nx.draw_networkx_edges(g, layout, arrowstyle=arrowstyle,
                                   arrowsize=arrowsize, edge_color=edge_color,
                                   width=edge_weight_list, ax=ax,
                                   **edge_kwargs
                                  )
    nodes = nx.draw_networkx_nodes(g, layout, ax=ax,
                                   node_size=node_size,
                                   node_color=node_color_list,
                                   **node_kwargs
                                  )
    #fig.set_frameon(False)
    _ = ax.axis('off')
    return g, (fig, ax)
