import scipy as sp
import pandas as pd
import networkx as nx
import palettable, warnings
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, r2_score
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error

sns.set_context("paper", font_scale=2)

def obtain_metric_classification(y_true, y_pred, y_score):
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

def obtain_metric_regression(y_true, y_pred):
    ''' for regression metrics, set errors to be negative (to pick the largest ones) '''
    r2 = r2_score(y_true, y_pred)
    mae_median = -median_absolute_error(y_true, y_pred)
    mae = -mean_absolute_error(y_true, y_pred)
    mse = -mean_squared_error(y_true, y_pred)

    d = {'r2': r2, 'mae_median': mae_median,
         'mae': mae, 'mse': mse}

    return pd.Series(d)


def visualize_cv(performance_file,
                 split_version,
                 validation_metric='auc_micro',
                 color_list=None,
                 legend_title_size=15,
                 metric_name=None,
                 fp_kwargs = {'aspect':2.7, 'size':2.5},
                 legend_kwargs={'loc': 8}
                ):
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
        fp_kwargs, legend_kwargs : dict
            args for factorplot and legend options
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
                        palette=color_list, row='variable',
                        row_order=['train_r2', validation_metric],
                        hue='lambda_', kind='point', sharey=False,
                        legend=False, **fp_kwargs,
                        #legend_out=True
                       )

    fp.axes[0,0].set_title('Training Performance', fontdict={'size': 14})
    fp.axes[1,0].set_title('Validation Performance', fontdict={'size': 14})
    fp.axes[0,0].set_ylabel('$R^2$\non training', fontdict={'size': 12})
    if metric_name is None:
        metric_name = validation_metric
    fp.axes[1,0].set_ylabel('{}\non heldout'.format(metric_name), fontdict={'size': 12})

    fp.axes[1,0].plot([best_x-1], [best_y], 'o', mfc='none', ms=20, mew=2, color='k')
    fp.axes[1,0].axhline(best_y, xmax=(best_x)/10, color='k', lw=2, ls='--')
    fp.axes[1,0].axvline(best_x-1, ymax=0.9, color='k', lw=2, ls='--')

    # https://stackoverflow.com/a/40910102/3346210
    handles = fp._legend_data.values()
    labels = fp._legend_data.keys()
    leg = fp.fig.legend(handles=handles, labels=labels,
                        **legend_kwargs
                       )
    leg.get_title().set_fontsize(legend_title_size)
    fp.fig.subplots_adjust(top=0.9,right=0.8)
    fp.fig.legend

    fp.set_xlabels('Features Selected', fontdict={'size': 12})
    return fp, best_x, best_y, best_lambda_, split_version

def visualize_s3d_steps(model_folder, figsize=(8,7), color_list=None, bar_alpha=1,
                        selectd_lw=4, selected_ls='-', selected_lc='k',
                        highlight_other=True, max_features=None,
                        other_lw=2, other_ls='--', other_lc='k'):
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
        bar_alpha : float
            alpha level of bars (0-1)
        highlight_other : bool
            whether or not to highlight features with equal contribution to $R^2$
        max_features : int
           if an integer, pick top `max_features` features at step 1 to be present in the bar chart and drop all others. if none (default), list all features.
        {selected,other}_{lw,ls,lc} : float/str
            line width/style/color for bar outlines
    '''
    if not model_folder.endswith('/'):
        model_folder += '/'
    df = pd.read_csv(model_folder+'R2improvements.csv')
    ## read in the selected ones
    selected_feature_arr = pd.read_csv(model_folder+'levels.csv')['best_feature'].values
    df = df.T.sort_values(0).T
    if max_features is not None:
        if max_features < selected_feature_arr.size:
            warnings.warn('max_features auto set to the size of selected features\n(change from {} to {})'.format(max_features,
                                                                                                                  selected_feature_arr.size))
            max_features = selected_feature_arr.size
        elif max_features > df.shape[1]:
            warnings.warn('max_features ({}) corrected to the total number of features ({})'.format(max_features, df.shape[1]))
            max_features = df.shape[1]
        ## pick the rest of the top features (at step 1) given that all selected features are included 
        other_size = max_features - selected_feature_arr.size
        #print('other_size', other_size)
        other_selected_feature = [col for col in df.columns[::-1] if col not in selected_feature_arr][:other_size]
        #print(other_selected_feature)
        df = df[selected_feature_arr.tolist()+other_selected_feature]
        #print(df)
        df = df.T.sort_values(0).T

    fig, ax = plt.subplots(figsize=figsize)
    left_base = 0
    width = 0.05
    y = pd.np.arange(df.shape[1])
    if color_list is None:
        num_color = max(3, df.shape[0])
        color_list = eval('palettable.colorbrewer.qualitative.Pastel1_'+str(num_color)+'.mpl_colors')
    elif len(color_list) < df.shape[0]:
        raise ValueError('color_list does not have enough colors ({}) for lambdas ({})'\
                          .format(len(color_list), df.shape[0]))
    ## highlight the x labels as well
    for i in range(df.shape[0]):
        ax.axvline(x=left_base, linestyle='--', color='k')
        #df.loc[i].plot(kind='barh', color=color_list[i], ax=ax, left=left_base, label='Step %d'%(i+1))
        i_bar = ax.barh(y, df.loc[i].values, color=color_list[i],
                        left=left_base, label='Step %d'%(i+1), alpha=bar_alpha)
        ## highlihght the highest bar (the selected one)
        i_series = df.loc[i]
        i_max = pd.np.argwhere(i_series.index==selected_feature_arr[i]).item()
        i_bar.patches[i_max].set_linewidth(selectd_lw)
        i_bar.patches[i_max].set_edgecolor(selected_lc)
        i_bar.patches[i_max].set_linestyle(selected_ls)
        left_base += df.loc[i].max()
        ## there may be multiple ones that match the max value
        if not highlight_other:
            continue
        max_val = i_series.max()
        i_max_arr = pd.np.argwhere(i_series==max_val).flatten()
        #print(i_max_arr)
        for i_m in i_max_arr:
            if i_m == i_max:
                continue
            i_bar.patches[i_m].set_linewidth(other_lw)
            i_bar.patches[i_m].set_edgecolor(other_lc)
            i_bar.patches[i_m].set_linestyle(other_ls)

    ax.set_xlabel(r'$R^2$')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_yticks(y)
    ax.set_yticklabels(df.columns)
    _ = ax.legend(loc='upper center', fancybox=True, shadow=True,
                  prop={'size': 12},
                  ncol=df.shape[0],
                  bbox_to_anchor=(0.5, 1.1))
    ## finally, remove legend patches outlines
    for legend_h in ax.legend_.legendHandles:
        legend_h.set_linewidth('0')
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
                #split[1] = 0.1
                pass
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
    N_masked = pd.np.ma.masked_where((Ns_mesh == 0), Ns_mesh)
    pred_masked = ((intensity_mesh >= thres) & (Ns_mesh >0)).astype(int)
    intensity_masked = pd.np.ma.masked_where((Ns_mesh==0)|(intensity_mesh<=0),
                                             intensity_mesh)
    return splits_at_dim, N_masked, intensity_masked, pred_masked, chosen_features[:dim]

def visualize_s3d_model(dim, splits_at_dim, cmap,
                        masked_arr, cbar_label,
                        chosen_features=None,
                        xscale='log', yscale='log',
                        xlab_x=None, xlab_y=None, norm_func=None,
                        ylab_x=None, ylab_y=None, scale=1,
                        fontsize=15, unit_w=3.3, unit_h=2.4,
                        xbins_lab_decimal=2, ybins_lab_decimal=2,
                        cb_kwargs={'aspect': 15},
                        cb_label_kwargs={'labelpad': 30, 'rotation': 270}
                        ):
    if dim == 1:
        msg = 'dim must be 2, 3, or 4 to use this function\n'
        msg += 'if you want to plot 1d visualization, use `visualization_s3d_model_1d`'
        raise ValueError(msg)
    nrows = ncols = 1
    if dim == 3:
        ncols = len(splits_at_dim[0])-1
    if dim == 4:
        nrows = len(splits_at_dim[0])-1
        ncols = len(splits_at_dim[1])-1

    figsize = (ncols*scale*unit_w, nrows*unit_h*scale)
    fig, ax_arr = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                               sharey=True, sharex=True)
    ax_arr = pd.np.reshape(ax_arr, (nrows, ncols))

    norm = None
    if norm_func is not None:
        try:
            norm = norm_func(masked_arr.min(), masked_arr.max())
        except:
            warnings.warn('norm_func may be inappropriate. try another one.\nnow no normalization is applied.')

    for i in range(nrows):
        for j in range(ncols):
            ax = ax_arr[i, j]
            if dim == 2:
                mesh_map = masked_arr
            elif dim == 3:
                mesh_map = masked_arr[j,:,:]
            else:
                mesh_map = masked_arr[nrows-i-1,j,:]
            im = ax.pcolormesh(splits_at_dim[dim-2], splits_at_dim[dim-1],
                               mesh_map.T, cmap=cmap,
                               vmin=masked_arr.min(),
                               vmax=masked_arr.max(),
                               norm=norm
                              )
            ax.set_xscale(xscale)
            ax.set_xlim(max(1, min(splits_at_dim[dim-2])), max(splits_at_dim[dim-2]))
            ax.set_yscale(yscale)
            ax.set_ylim(max(1, min(splits_at_dim[dim-1])), max(splits_at_dim[dim-1]))

            if dim > 2 and i == nrows-1:
                xlab_str = '{}'.format(chosen_features[dim-2])
                if j == 0:
                    start_paranthesis = '['
                else:
                    start_paranthesis = '('
                #xlab_str += '\n${}{}, {}]$'.format(start_paranthesis,
                #                                   splits_at_dim[dim-3][j],
                #                                   splits_at_dim[dim-3][j+1])
                #print(xlab_str)
                xlab_str += '\n${0}{1:.{2}f}, {3:.{4}f}]$'.format(start_paranthesis,
                                                                  splits_at_dim[dim-3][j],
                                                                  xbins_lab_decimal,
                                                                  splits_at_dim[dim-3][j+1],
                                                                  xbins_lab_decimal,
                                                                 )
                ax.set_xlabel(xlab_str, size=fontsize)
            elif dim == 2:
                ax.set_xlabel(chosen_features[0], size=fontsize)

            if j == 0:
                if dim == 4:
                    if i == nrows-1:
                        start_paranthesis = '['
                    else:
                        start_paranthesis = '('
                    ylab_str = '${0}{1:.{2}f}, {3:.{4}f}]$\n'.format(start_paranthesis,
                                                                     splits_at_dim[0][nrows-i-1],
                                                                     ybins_lab_decimal,
                                                                     splits_at_dim[0][nrows-i],
                                                                     ybins_lab_decimal,
                                                                    )
                                                      #round(splits_at_dim[0][nrows-i-1], ybins_lab_decimal),
                                                      #round(splits_at_dim[0][nrows-i], ybins_lab_decimal))
                    #print(ylab_str) 
                    ylab_str += '{}'.format(chosen_features[dim-1])
                else:
                    ylab_str = '{}'.format(chosen_features[dim-1])
                ax.set_ylabel(ylab_str, size=fontsize)

    if dim == 3:
        if xlab_x is None:
            xlab_x = .33*scale
        if xlab_y is None:
            xlab_y = -0.4/scale
        fig.text(x=xlab_x, y=xlab_y, s=chosen_features[dim-3], size=fontsize)
        #fig.suptitle(chosen_features[dim-3], size=fontsize, y=-0.25, x=0.45)
    elif dim == 4:
        if xlab_x is None:
            xlab_x = .38*scale
        if xlab_y is None:
            xlab_y = 0.01/scale
        fig.text(x=xlab_x, y=xlab_y, s=chosen_features[dim-3], size=fontsize)

        if ylab_x is None:
            ylab_x = .008*scale
        if ylab_y is None:
            ylab_y = 0.6/scale
        fig.text(x=ylab_x, y=ylab_y, s=chosen_features[0], size=fontsize, rotation=90)

    # colorbar
    cb = fig.colorbar(im, ax=ax_arr.ravel().tolist(),
                      pad=0.1/(dim+.5), **cb_kwargs)
    cb.set_label(cbar_label, **cb_label_kwargs)
    return fig, ax_arr

def visualize_s3d_model_1d(splits_at_dim, masked_arr,
                           xlab=None, ylab=None,
                           figsize=(8,6), xscale='log', yscale='log',
                           vlines_kwargs={'linestyles': ':', 'linewidth': .3, 'color': 'k'},
                           hlines_kwargs={'linestyles': '-', 'linewidth': 2, 'color': 'gray'},
                          ):
    ''' visualize s3d model using the top feature using line chart'''
    fig, ax = plt.subplots(figsize=figsize)
    splits = splits_at_dim[0]
    assert len(splits)-1 == masked_arr.size
    for i, val_i in enumerate(masked_arr):
        ax.hlines(y=val_i, xmin=splits[i],
                  xmax=splits[i+1],
                  **hlines_kwargs,
                 )
        if i == masked_arr.size-1:
            ymax_val = masked_arr[0]
        else:
            ymax_val = masked_arr[i+1]
        ax.vlines(x=splits[i+1], ymin=val_i,
                  ymax=ymax_val,
                  **vlines_kwargs,
                 )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    return fig, ax

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

    color_dict = nx.get_node_attributes(g, 'color')
    node_color_list = [color_dict[u] for u in g.nodes()]

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

def find_best_param(performance_file, validation_metric):
    df = pd.read_csv(performance_file)
    id_vars_list = ['split_version', 'lambda_', 'num_features']
    cv_df = df.groupby(id_vars_list).mean()[[validation_metric, 'train_r2']]

    ## reshape the data

    cv_df = cv_df.reset_index().melt(id_vars=id_vars_list,
                                     value_vars=[validation_metric, 'train_r2'])

    ## group by each `split_version` to obtain the best `lambda_` and `num_features`
    param_df = list()
    for s_ver, split_cv_df in cv_df.groupby('split_version'):
        split_cv_df = split_cv_df.set_index(id_vars_list[1:])
        best_lambda_, best_n_f = split_cv_df.query("variable==@validation_metric").value.idxmax()
        best_value = split_cv_df.query("variable==@validation_metric").value.max()
        param_df.append([s_ver, best_lambda_, best_n_f, best_value, validation_metric])
    param_df = pd.DataFrame(param_df, columns=['split_version', 'lambda_', 'num_features', 'best_value', 'metric'])
    return param_df

'''
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
'''
