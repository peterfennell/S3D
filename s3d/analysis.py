import scipy as sp
import pandas as pd
import networkx as nx
import palettable, warnings
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, r2_score
# from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error



def visualize_s3d_steps(model, figsize=(8,7), color_list=None, bar_alpha=1,
            selectd_lw=4, selected_ls='-', selected_lc='k',
            highlight_other=True, max_features=None,
            other_lw=2, other_ls='--', other_lc='k'):
  ''' visualize the increment of r-squared of s3d model
    Parameters
    ----------
    model : instance of an S3D object
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
  df = model.r2_improvements
  selected_feature_arr = model.selected_features
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
      ## do not limit x/y axes values
      #ax.set_xlim(max(1, min(splits_at_dim[dim-2])), max(splits_at_dim[dim-2]))
      ax.set_yscale(yscale)
      #ax.set_ylim(max(1, min(splits_at_dim[dim-1])), max(splits_at_dim[dim-1]))

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
  return fig, ax_arr, cb

