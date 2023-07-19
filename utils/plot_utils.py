import numpy as np
import torch
import matplotlib.pyplot as plt

## choose figure or poster defaults
#fig_defaults = 'poster'                                 # poster defaults
#fig_defaults = 'presentation'                           # presentation defaults
fig_defaults = 'paper'                                  # paper figure defaults

if fig_defaults == 'paper':
    ####### paper figure defaults
    label_fontsize = 6 # pt
    ticklabel_fontsize = 6 # pt
    plot_linewidth = 0.5 # pt
    linewidth = 0.5
    axes_linewidth = 0.5
    marker_size = 3.0 # markersize=<...>
    cap_size = 2.0 # for errorbar caps, capsize=<...>
    columnwidth = 85/25.4 # inches
    twocolumnwidth = 174/25.4 # inches
    linfig_height = columnwidth*2.0/3.0
    fig_dpi = 300
elif fig_defaults == 'poster':
    ####### poster defaults
    label_fontsize = 12 # pt
    ticklabel_fontsize = 8 # pt
    plot_linewidth = 1.0 # pt
    linewidth = 1.0
    axes_linewidth = 1.0
    marker_size = 3.0
    cap_size = 2.0 # for errorbar caps
    columnwidth = 4 # inches
    linfig_height = columnwidth*2.0/3.0
else:
    ####### presentation defaults for screenshot
    label_fontsize = 20 # pt
    ticklabel_fontsize = 8 # pt
    plot_linewidth = 0.5 # pt
    linewidth = 1.0#0.5
    axes_linewidth = 0.5
    marker_size = 3.0 # markersize=<...>
    cap_size = 2.0 # for errorbar caps, capsize=<...>
    columnwidth = 85/25.4 # inches
    twocolumnwidth = 174/25.4 # inches
    linfig_height = columnwidth*2.0/3.0
    fig_dpi = 300

def get_labeled_axes(labels_no = 4, figsize = [15/2.53, 8/2.53]):
    labels_dict = {2: [['A', 'B']], 3: ['A', 'B', 'C'], 4: [['A', 'B'], ['C', 'D']], 6: [['A','B','C'], ['D', 'E', 'F']]}
    fig, axes = plt.subplot_mosaic(labels_dict[labels_no],
                                constrained_layout=False, figsize = figsize)
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-33/72, -4/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')
    return fig, axes

def stats(var, var_name=None):
    if type(var) == type([]): # if a list
        var = np.array(var)
    elif type(var) == type(np.array([])):
        pass #if already a numpy array, just keep going.
    else: #assume torch tensor
        pass
        # var = var.detach().cpu().numpy()
    if var_name:
        print(var_name, ':')   
    out = ('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {}'.format(var.mean(), var.var(), var.min(), var.max(),np.linalg.norm(var) ))
    print(out)
    return (out)

def draw_context_span(ax, switches, color='tab:red'):
    for s_i in range(0, len(switches)-1, 2):
        ax.axvspan(switches[s_i], switches[s_i+1], color=color, alpha=0.1)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def set_tick_widths(ax,tick_width):
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)


def axes_off(ax,x=True,y=True,xlines=False,ylines=False):
    ''' True is to turn things off, False to keep them on!
        x,y are for ticklabels, xlines,ylines are for ticks'''
    if x:
        for xlabel_i in ax.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
    if y:
        for xlabel_i in ax.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
    if xlines:
        for tick in ax.get_xticklines():
            tick.set_visible(False)
    if ylines:
        for tick in ax.get_yticklines():
            tick.set_visible(False)

def axes_labels(ax,xtext,ytext,adjustpos=False,fontsize=ticklabel_fontsize,xpad=None,ypad=None):
    ax.set_xlabel(xtext,fontsize=label_fontsize,labelpad=xpad)
    # increase xticks text sizes
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    ax.set_ylabel(ytext,fontsize=label_fontsize,labelpad=ypad)
    # increase yticks text sizes
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
    if adjustpos:
        ## [left,bottom,width,height]
        ax.set_position([0.135,0.125,0.84,0.75])
    set_tick_widths(ax,axes_linewidth)

    for loc, spine in ax.spines.items(): # items() returns [(key,value),...]
        spine.set_linewidth(axes_linewidth)



def fig_clip_off(fig):
    ## clipping off for all objects in this fig
    for o in fig.findobj():
        o.set_clip_on(False)

