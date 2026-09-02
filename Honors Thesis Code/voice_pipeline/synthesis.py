"""
Stage D: shape descriptors + the two-cluster check across the whole feature
battery, plus plotting.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def shape_descriptors(burst_stat: float, accumulator_slope: float) -> dict:
    return {'burst_effect': burst_stat, 'accumulator_slope': accumulator_slope}


def two_cluster_check(descriptor_table, feature_names):
    """descriptor_table: array of shape (n_features, 2) = [burst_effect, accumulator_slope].
    Returns cluster labels + silhouette score for k=2, so we can say whether
    the features actually separate into two families or just spread out
    along a continuum."""
    X = np.asarray(descriptor_table, dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    X_valid = X[valid]
    names_valid = [n for n, v in zip(feature_names, valid) if v]

    if len(X_valid) < 4:
        return {'labels': {}, 'silhouette': np.nan, 'note': 'too few valid features to cluster'}

    Xs = StandardScaler().fit_transform(X_valid)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    labels = {name: int(lab) for name, lab in zip(names_valid, km.labels_)}
    return {'labels': labels, 'silhouette': float(sil), 'centers': km.cluster_centers_.tolist(),
            'note': ('Silhouette > ~0.5 suggests genuinely separated clusters; near 0 suggests '
                      'features spread on a continuum rather than falling into two families.')}


# Features whose raw value is mechanically sample-size-dependent (shrinks as
# window size grows, independent of any authorial signal) -- their CUMULATIVE
# panel will show a steep early move that is largely/entirely this artifact,
# not voice. The LOCAL panel (fixed window size throughout) and the actual
# Stage C statistical tests are unaffected -- they only ever use local scores.
SAMPLE_SIZE_SENSITIVE = {'ttr', 'hapax_rate'}


def plot_feature_curve(feature_name, local_scores, cumulative_scores, out_path, burst_p=None, accum_p=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(range(len(local_scores)), local_scores, marker='o', color='#3b6ea5')
    axes[0].axhline(0, color='gray', linewidth=0.7, linestyle='--')
    axes[0].set_title(f'{feature_name} — local windows')
    axes[0].set_xlabel('window index (start → end of book)')
    axes[0].set_ylabel('distinctiveness score')

    axes[1].plot(range(len(cumulative_scores)), cumulative_scores, marker='o', color='#a53b3b')
    axes[1].axhline(0, color='gray', linewidth=0.7, linestyle='--')
    cum_title = f'{feature_name} — cumulative windows'
    if feature_name in SAMPLE_SIZE_SENSITIVE:
        cum_title += '  [sample-size artifact -- see README]'
    axes[1].set_title(cum_title, fontsize=9.5 if feature_name in SAMPLE_SIZE_SENSITIVE else 11)
    axes[1].set_xlabel('cumulative window index')
    axes[1].set_ylabel('distinctiveness score')

    subtitle = []
    if burst_p is not None:
        subtitle.append(f'burst p={burst_p:.3f}')
    if accum_p is not None:
        subtitle.append(f'accumulator p={accum_p:.3f}')
    if subtitle:
        fig.suptitle('  |  '.join(subtitle), fontsize=9, y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_scatter(descriptor_table, feature_names, labels, out_path):
    X = np.asarray(descriptor_table, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {0: '#3b6ea5', 1: '#a53b3b'}
    for i, name in enumerate(feature_names):
        lab = labels.get(name)
        c = colors.get(lab, '#888888')
        ax.scatter(X[i, 0], X[i, 1], color=c, s=45)
        ax.annotate(name, (X[i, 0], X[i, 1]), fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_xlabel('burst effect (opening minus rest)')
    ax.set_ylabel('accumulator slope (trend across book)')
    ax.set_title('Feature battery: burst vs. accumulator shape')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
