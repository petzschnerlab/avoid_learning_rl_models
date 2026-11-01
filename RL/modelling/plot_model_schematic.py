"""
Schematic contrasting the reinforcement learning models (reviewer request).

Panel a: where context enters each model (Q-Learning reference, Relative, Advantage, Hybrid).
Panel b: the transfer-phase predictions they share — learned stimulus values move along
         one continuum from absolute to relative valuation (exact for the Advantage model
         at asymptote; Relative and Hybrid map onto the same continuum approximately).

Run from the repo root:
    python RL/plot_model_schematic.py

Output: RL/plots/model_schematic_mechanisms.png/.svg/.pdf  (panel a)
        RL/plots/model_schematic_continuum.png/.svg/.pdf   (panel b)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RL_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(RL_DIR))

if 'Helvetica' in set(f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['svg.fonttype'] = 'none'  # keep text editable in the SVG

# Colours (from Plotting.get_colors)
HR, LR, LP, HP = '#095086', '#9BD2F2', '#ECA6A6', '#B00000'
CONTEXT = '#3E7B27'        # context / relative component
CONTEXT_FILL = '#E6F0DF'
BASE_EDGE = '#555555'
BASE_FILL = '#F2F2F2'
TEXT = '#222222'


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, text, context=False, fontsize=12, bold=False):
    """Rounded box centred at (x, y)."""
    edge = CONTEXT if context else BASE_EDGE
    fill = CONTEXT_FILL if context else BASE_FILL
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.25',
                                linewidth=2 if context else 1.2,
                                edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=TEXT,
            fontweight='bold' if bold else 'normal', zorder=3, linespacing=1.35)


def arrow(ax, start, end, context=False, rad=0.0, style='-|>', lw=None):
    colour = CONTEXT if context else BASE_EDGE
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                                 linewidth=lw or (2 if context else 1.4), color=colour,
                                 connectionstyle=f'arc3,rad={rad}', zorder=1,
                                 shrinkA=2, shrinkB=2))


def bypass(ax):
    """Outcome -> prediction error path routed around the left of the context box."""
    x_out, x_left = 5 - W / 2, 0.55
    ax.plot([x_out, x_left, x_left], [Y_OUT, Y_OUT, Y_PE], color=BASE_EDGE, linewidth=1.4, zorder=1,
            solid_joinstyle='round')
    arrow(ax, (x_left, Y_PE), (x_out, Y_PE))


def column_header(ax, title, subtitle, context=False):
    ax.text(5, 15.4, title, ha='center', va='center', fontsize=16, fontweight='bold', color=TEXT)
    ax.text(5, 14.6, subtitle, ha='center', va='center', fontsize=12, style='italic',
            color=CONTEXT if context else BASE_EDGE)


def footer(ax, text, context=True):
    ax.text(5, 0.35, text, ha='center', va='center', fontsize=12,
            color=CONTEXT if context else BASE_EDGE, linespacing=1.35)


def setup(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')


# Shared vertical positions
Y_OUT, Y_CTX, Y_PE, Y_VAL, Y_CHOICE = 13.0, 10.4, 7.6, 5.0, 2.4
W, H = 7.4, 1.5


def draw_qlearning(ax):
    setup(ax)
    column_header(ax, 'Q-Learning', 'reference: absolute value only')
    box(ax, 5, Y_OUT, W, H, 'Outcomes\n(chosen and unchosen)')
    box(ax, 5, Y_PE, W, H, r'Prediction error' + '\n' + r'$\delta = R - Q$')
    box(ax, 5, Y_VAL, W, H, 'Value  $Q$\n(expected outcome)')
    box(ax, 5, Y_CHOICE, W, H, 'Choice\n(softmax over $Q$)')
    arrow(ax, (5, Y_OUT - H / 2), (5, Y_PE + H / 2))
    arrow(ax, (5, Y_PE - H / 2), (5, Y_VAL + H / 2))
    arrow(ax, (5, Y_VAL - H / 2), (5, Y_CHOICE + H / 2))
    footer(ax, 'No context: values reflect\nwhat each option pays overall', context=False)


def draw_actorcritic(ax):
    setup(ax)
    column_header(ax, 'Actor-Critic', 'reference: relative value only', context=True)
    box(ax, 5, Y_OUT, W, H, 'Outcomes\n(chosen and unchosen)')
    box(ax, 5, Y_CTX, W, H, 'Critic: state value  $V$\nlearned from chosen outcome', context=True)
    box(ax, 5, Y_PE, W, H, r'Prediction error' + '\n' + r'$\delta = R - V$')
    box(ax, 5, Y_VAL, W, H, 'Actor: preference  $w$\n(which was better in its pair)', context=True)
    box(ax, 5, Y_CHOICE, W, H, 'Choice\n(softmax over $w$)')
    arrow(ax, (5, Y_OUT - H / 2), (5, Y_CTX + H / 2), context=True)
    arrow(ax, (5, Y_CTX - H / 2), (5, Y_PE + H / 2), context=True)
    bypass(ax)
    arrow(ax, (5, Y_PE - H / 2), (5, Y_VAL + H / 2))
    arrow(ax, (5, Y_VAL - H / 2), (5, Y_CHOICE + H / 2))
    footer(ax, 'Only context: preferences reflect\nwhich option was better in its pair')


def draw_relative(ax):
    setup(ax)
    column_header(ax, 'Relative', 'context enters learning (learned)', context=True)
    box(ax, 5, Y_OUT, W, H, 'Outcomes\n(chosen and unchosen)')
    box(ax, 5, Y_CTX, W, H, 'Context value  $V$\nlearned from average outcome', context=True)
    box(ax, 5, Y_PE, W, H, r'Prediction error' + '\n' + r'$\delta = R - V - Q$')
    box(ax, 5, Y_VAL, W, H, 'Value  $Q$\n(relative to context)')
    box(ax, 5, Y_CHOICE, W, H, 'Choice\n(softmax over $Q$)')
    arrow(ax, (5, Y_OUT - H / 2), (5, Y_CTX + H / 2), context=True)
    arrow(ax, (5, Y_CTX - H / 2), (5, Y_PE + H / 2), context=True)
    bypass(ax)  # outcomes also feed the prediction error directly
    arrow(ax, (5, Y_PE - H / 2), (5, Y_VAL + H / 2))
    arrow(ax, (5, Y_VAL - H / 2), (5, Y_CHOICE + H / 2))
    # self-update loop on V
    arrow(ax, (8.7, Y_CTX + 0.4), (8.7, Y_CTX - 0.4), context=True, rad=-1.6, lw=1.6)
    ax.text(9.55, Y_CTX, r'$\alpha_v$', ha='left', va='center', fontsize=12, color=CONTEXT)
    footer(ax, 'More relative as $V$ is learned\n(higher contextual learning rate $\\alpha_v$)')


def draw_advantage(ax):
    setup(ax)
    column_header(ax, 'Advantage', 'context enters learning (each trial)', context=True)
    box(ax, 5, Y_OUT, W, H, 'Outcomes\n(chosen and unchosen)')
    box(ax, 5, Y_CTX, W, H, 'Trial average  $m \\cdot \\bar{R}$\ncomputed directly, not learned', context=True)
    box(ax, 5, Y_PE, W, H, r'Prediction error' + '\n' + r'$\delta = R - m\bar{R} - Q$')
    box(ax, 5, Y_VAL, W, H, 'Value  $Q$\n(relative to context)')
    box(ax, 5, Y_CHOICE, W, H, 'Choice\n(softmax over $Q$)')
    arrow(ax, (5, Y_OUT - H / 2), (5, Y_CTX + H / 2), context=True)
    arrow(ax, (5, Y_CTX - H / 2), (5, Y_PE + H / 2), context=True)
    bypass(ax)
    arrow(ax, (5, Y_PE - H / 2), (5, Y_VAL + H / 2))
    arrow(ax, (5, Y_VAL - H / 2), (5, Y_CHOICE + H / 2))
    footer(ax, 'More relative as weighting $m \\to 1$\n($m = 0$ is Q-Learning)')


def draw_hybrid(ax):
    setup(ax)
    column_header(ax, 'Hybrid', 'context enters at the decision', context=True)
    box(ax, 5, Y_OUT, W, H, 'Outcomes\n(chosen and unchosen)')
    sw = 4.1
    box(ax, 2.6, (Y_CTX + Y_PE) / 2 + 0.3, sw, 3.4,
        'Q-learner\n\nabsolute\nvalue  $Q$', fontsize=12)
    box(ax, 7.4, (Y_CTX + Y_PE) / 2 + 0.3, sw, 3.4,
        'Actor-critic\n\nrelative\npreference  $w$', context=True, fontsize=12)
    box(ax, 5, Y_VAL, W, H, 'Mixture\n$H = c\\,Q + (1-c)\\,w$', context=True)
    box(ax, 5, Y_CHOICE, W, H, 'Choice\n(softmax over $H$)')
    top = (Y_CTX + Y_PE) / 2 + 0.3 + 1.7
    bottom = (Y_CTX + Y_PE) / 2 + 0.3 - 1.7
    arrow(ax, (3.6, Y_OUT - H / 2), (2.6, top))
    arrow(ax, (6.4, Y_OUT - H / 2), (7.4, top), context=True)
    arrow(ax, (2.6, bottom), (3.8, Y_VAL + H / 2))
    arrow(ax, (7.4, bottom), (6.2, Y_VAL + H / 2), context=True)
    arrow(ax, (5, Y_VAL - H / 2), (5, Y_CHOICE + H / 2))
    footer(ax, 'More relative as mixing $c \\to 0$\n(two systems learn in parallel)')


def draw_continuum(ax):
    """Asymptotic Advantage values: v = EV - m * context mean (context mean = +/-0.5)."""
    m = np.linspace(0, 1, 101)
    lines = {'High Reward (HR)': (0.75 - 0.5 * m, HR),
             'Low Reward (LR)': (0.25 - 0.5 * m, LR),
             'Low Punish (LP)': (-0.25 + 0.5 * m, LP),
             'High Punish (HP)': (-0.75 + 0.5 * m, HP)}
    for label, (v, c) in lines.items():
        ax.plot(m, v, color=c, linewidth=4, label=label, solid_capstyle='round')

    # LR = LP crossing
    ax.axvline(0.5, color='#999999', linestyle=':', linewidth=1.5, zorder=0)
    ax.scatter([0.5], [0], s=90, color='white', edgecolor=TEXT, linewidth=1.5, zorder=5)
    ax.annotate('LR = LP: no preference\n(m = 0.5)', xy=(0.5, 0), xytext=(0.17, -0.42),
                fontsize=12, color=TEXT, arrowprops=dict(arrowstyle='-', color='#999999'))

    # Constant within-pair gap
    ax.annotate('', xy=(0.06, 0.72), xytext=(0.06, 0.22),
                arrowprops=dict(arrowstyle='<->', color=TEXT, linewidth=1.2))
    ax.text(0.085, 0.47, 'within-pair gap\nnever changes', fontsize=11, va='center', color=TEXT)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Absolute\n(expected value)', 'Equal\nweighting', 'Fully relative\n(within-pair rank)'])
    ax.set_ylabel('Learned value')
    ax.set_yticks([-0.75, -0.25, 0.25, 0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=12)

    # Parameter mapping under the axis
    rows = [('Reference', 'Q-Learning  →  Actor-Critic'),
            ('Advantage', '$m$:  0  →  1'),
            ('Relative', '$V$ learned:  none  →  fully'),
            ('Hybrid', '$c$:  1  →  0')]
    for i, (name, txt) in enumerate(rows):
        y = -0.2 - 0.065 * i
        colour = BASE_EDGE if i == 0 else CONTEXT
        ax.text(-0.02, y, name, transform=ax.transAxes, ha='right', va='center',
                fontsize=12, fontweight='bold', color=colour)
        ax.text(0.5, y, txt, transform=ax.transAxes, ha='center', va='center', fontsize=12, color=colour)
    ax.text(0.5, -0.2 - 0.065 * 4, 'Values: Advantage model after learning. Relative and Hybrid map onto '
            'the same axis approximately.', transform=ax.transAxes, ha='center', va='center',
            fontsize=10.5, style='italic', color=BASE_EDGE)


def draw_takeaways(ax):
    ax.axis('off')
    items = [
        ('Learning phase', 'Choices are always within a pair. The context term is\n'
                           'shared by both options and cancels, so the learning\n'
                           'phase barely distinguishes the models.'),
        ('Transfer phase', 'Options from different contexts meet. The models\n'
                           'differ in how far values are pulled toward the\n'
                           'within-pair rank, revealed by LR vs LP choices.'),
        ('Implication', 'The three mixture models move behaviour along\nthe same continuum, between the two references.\n'
                        'They differ in mechanism (where context enters),\n'
                        'not in the choices they predict for this task.'),
    ]
    y = 0.95
    for title, body in items:
        ax.text(0, y, title, fontsize=13, fontweight='bold', color=TEXT, va='top', transform=ax.transAxes)
        ax.text(0, y - 0.075, body, fontsize=12, color=TEXT, va='top', linespacing=1.4, transform=ax.transAxes)
        y -= 0.34


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def save(fig, name):
    os.makedirs('RL/plots', exist_ok=True)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(f'RL/plots/{name}.{ext}', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print(f'Saved RL/plots/{name}.png/.svg/.pdf')


if __name__ == '__main__':

    # ---- Figure A: where context enters each model ---- #
    fig_a, axes = plt.subplots(1, 5, figsize=(22, 9.5), gridspec_kw={'wspace': 0.08})
    for ax, draw in zip(axes, [draw_qlearning, draw_actorcritic, draw_relative, draw_advantage, draw_hybrid]):
        draw(ax)
    fig_a.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03)

    # Divider between reference models and mixture models
    a1, a2 = axes[1].get_position(), axes[2].get_position()
    x_div = (a1.x1 + a2.x0) / 2
    fig_a.add_artist(plt.Line2D([x_div, x_div], [a1.y0 + 0.04, a1.y1 - 0.08], color='#CCCCCC',
                                linewidth=1.2, linestyle='--', transform=fig_a.transFigure))

    # Colour key
    from matplotlib.patches import Patch
    key = [Patch(facecolor=BASE_FILL, edgecolor=BASE_EDGE, linewidth=1.2,
                 label='Standard learning step (shared by all models)'),
           Patch(facecolor=CONTEXT_FILL, edgecolor=CONTEXT, linewidth=2,
                 label='Where context enters (makes values relative)')]
    fig_a.legend(handles=key, loc='upper right', bbox_to_anchor=(0.98, 0.985), ncol=2, frameon=False,
                 fontsize=13, handlelength=2.2, handleheight=1.3, columnspacing=2.5)
    save(fig_a, 'model_schematic_mechanisms')

    # ---- Figure B: shared transfer-phase prediction ---- #
    fig_b = plt.figure(figsize=(14, 9))
    ax_b = fig_b.add_subplot(111)
    draw_continuum(ax_b)
    fig_b.subplots_adjust(left=0.13, right=0.77, top=0.96, bottom=0.34)
    save(fig_b, 'model_schematic_continuum')
