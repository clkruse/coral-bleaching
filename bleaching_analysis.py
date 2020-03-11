import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_ind
from sklearn.manifold import TSNE

db = pd.read_json('raw_window_10_params_11.json')

variables = {
    "sst": {
        "min": 5,
        "max": 50
    },
    "chlor_a": {
        "min": 0,
        "max": 1
    },
    "velocity": {
        "min": 0,
        "max": 50
    },
    "salinity": {
        "min": 0,
        "max": 20
    },
    "wind_avhrr": {
        "min": 0,
        "max": 50
    },
    "cloud_transmission": {
        "min": -127,
        "max": 127
    }
}

# Convert directional component velocity into a velocity magnitude
magnitude = lambda u, v: math.sqrt(u ** 2  + v ** 2)
velocity_mag = []
for vel_u, vel_v in zip(db['raw_velocity_u'], db['raw_velocity_u']):
    mag = [magnitude(u, v) for u, v in zip(vel_u, vel_v)]
    velocity_mag.append(mag)
db['raw_velocity'] = velocity_mag
db = db.drop(columns = ['raw_velocity_u', 'raw_velocity_v'])

# Filter the raw input data
for variable in db.columns:
    if 'raw_' in variable:
        product = str.split(variable, 'raw_')[1]
        filtered_values = []
        for entry in db[variable]:
            for category in variables:
                if category in product:
                    min = variables[category]['min']
                    max = variables[category]['max']

            filtered_entry = [value for value in entry if value < max and value > min]
            filtered_values.append(filtered_entry)
        db[product] = filtered_values


# Merge the multiple sources of chlorophyll a
# Prefer seawifs, then modis terra, then modis aqua
db['chlor_a'] = db['chlor_a_seawifs'].fillna(db['chlor_a_terra']).fillna(db['chlor_a_aqua'])

# Merge the multiple sources of temperature
# Prefer OISST, then sst WHOI model, then AVHRR (AVHRR is probably good, but can't measure through clouds
db['sst'] = db['sst'].fillna(db['sst_whoi']).fillna(db['sst_avhrr'])

for variable in db.columns:
    if variable in variables:
        means, meds, mins, maxs, variances = [], [], [], [], []
        for entry in db[variable]:
            if len(entry) > 0:
                means.append(np.mean(entry))
                meds.append(np.median(entry))
                mins.append(np.min(entry))
                maxs.append(np.max(entry))
                variances.append(np.var(entry))
            else:
                means.append(np.NaN)
                meds.append(np.NaN)
                mins.append(np.NaN)
                maxs.append(np.NaN)
                variances.append(np.NaN)
        db['mean_' + variable] = means
        db['med_' + variable] = meds
        db['min_' + variable] = mins
        db['max_' + variable] = maxs
        db['variance_' + variable] = variances


stats = ['mean_', 'med_', 'min_', 'max_']
healthy = db[db['severity'] == 0]
bleaching = db[db['severity'] != 0]

for variable in variables:
    for stat in stats:
        pval = ttest_ind(healthy[stat + variable].dropna(), bleaching[stat + variable].dropna()).pvalue
        print(stat + variable, "healthy {0:.2f}, bleaching {1:.2f}, pval: {2:.6f}".format(healthy[stat + variable].dropna().mean(), bleaching[stat + variable].dropna().mean(), pval))


counter = 1
plt.figure(figsize=(24, 30))
for variable in variables:
    for stat in stats:
        mean_healthy = healthy[stat + variable].mean()
        var_healthy = math.sqrt(healthy[stat + variable].var())
        mean_bleaching = bleaching[stat + variable].mean()
        var_bleaching = math.sqrt(bleaching[stat + variable].var())
        plt.subplot(len(variables), len(stats), counter)
        plt.errorbar([0, 1], [mean_healthy, mean_bleaching], [var_healthy, var_bleaching], fmt='.', color='black')
        plt.bar(0, mean_healthy)
        plt.bar(1, mean_bleaching, color='r')
        plt.xticks([0, 1], ['Healthy', 'Bleaching'])
        #plt.legend(['Healthy', 'Bleaching'])
        title = str(stat + variable + "\nHealthy: {0:.1f}, Bleaching: {1:.1f}").format(mean_healthy, mean_bleaching)
        plt.title(title)
        counter += 1
plt.suptitle('Mean Values for each Paramter in the 10 Days before and after an Assessment\nValues from 2821 Bleaching Reefs and 900 Healthy Reefs', size=16, y=0.92)
plt.savefig('figures/Comparison of Mean Values - Window 10.png', dpi=300, bbox_inches='tight')
plt.show()


counter = 1
plt.figure(figsize=(24, 30))
for variable in variables:
    for stat in stats:
        val_healthy = healthy[stat + variable].dropna()
        val_bleaching = bleaching[stat + variable].dropna()
        min_val = np.min(pd.concat([val_bleaching, val_healthy]))
        max_val = np.max(pd.concat([val_bleaching, val_healthy]))
        xs = np.linspace(min_val, max_val, 200)

        density_healthy = gaussian_kde(val_healthy)
        normed_healthy = density_healthy(xs)/np.sum(density_healthy(xs))
        density_bleaching = gaussian_kde(val_bleaching)
        normed_bleaching = density_bleaching(xs)/np.sum(density_bleaching(xs))

        median_healthy = np.median(val_healthy)
        median_bleaching = np.median(val_bleaching)

        mean_healthy = np.mean(val_healthy)
        mean_bleaching = np.mean(val_bleaching)

        plt.subplot(len(variables), len(stats), counter)

        plt.plot(xs, normed_healthy, label='Healthy')
        plt.plot(xs, normed_bleaching, color='red', label='Bleaching')
        plt.vlines(median_healthy, 0, np.max([normed_healthy, normed_bleaching]), color='C0', alpha = 0.5, linestyle=':')
        plt.vlines(median_bleaching, 0, np.max([normed_healthy, normed_bleaching]), color='r', alpha = 0.5, linestyle=':', label='Median')

        plt.vlines(mean_healthy, 0, np.max([normed_healthy, normed_bleaching]), color='C0', alpha = 0.5)
        plt.vlines(mean_bleaching, 0, np.max([normed_healthy, normed_bleaching]), color='r', alpha = 0.5, label='Mean')

        plt.ylim(bottom=0)
        #plt.legend(['Healthy', 'Bleaching', 'Median', '_nolegend_', 'Mean'])
        plt.legend()
        title = str("Distribution of " + stat + variable + "\nMean Healthy: {0:.1f}, Bleaching: {1:.1f}").format(np.mean(val_healthy), np.mean(val_bleaching))
        plt.title(title)
        counter += 1
plt.suptitle('Distribution of Values for 10 Days before and after an Assessment\nValues from > 2500 Bleaching Reefs and >800 Healthy Reefs', size=16, y=0.92)
plt.savefig('figures/Distributions of Values - Window 10.png', dpi=300, bbox_inches='tight')
plt.show()


colors =['#2B2D42', '#8D99AE', '#D65959', '#C93434', '#AC0808']
ordered_severity = [0, -1, 1, 2, 3]
counter = 1
plt.figure(figsize=(24, 30))
for variable in variables:
    for stat in stats:
        plt.subplot(len(variables), len(stats), counter)
        for severity in ordered_severity:
            values =  db[db['severity'] == severity]
            mean_val = values[stat + variable].mean()
            var_val = math.sqrt(values[stat + variable].var())
            plt.errorbar(severity, mean_val, var_val, fmt='.', color='black')
            plt.bar(severity, mean_val, color=colors[severity+1])
            plt.xticks([-1, 0, 1, 2, 3], ['None', 'Unknown\n(bleaching)', 'Low', 'Mid', 'High'])
            #plt.legend(['Healthy', 'Bleaching'])
            title = str(stat + variable)
            plt.title(title)
        counter += 1
plt.suptitle("Averaged Parameter Value per Statistic Grouped by Severity", size=16, y=0.92)
plt.savefig('figures/Mean Value by Severity - Window 10.png', dpi=300, bbox_inches='tight')
plt.show()


ordered_severity = [0, -1, 1, 2, 3]
counter = 1
plt.figure(figsize=(24, 29))
for variable in variables:
    for stat in stats:
        plt.subplot(len(variables), len(stats), counter)
        data = []
        for severity in ordered_severity:
            values = db[db['severity'] == severity]
            data.append(values[stat + variable].dropna())

        box_plot = plt.boxplot(data, positions=[0, 1, 2, 3, 4],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": '#d1d1d1'})

        title = str(stat + variable)
        plt.title(title)
        plt.xticks([0, 1, 2, 3, 4], ['None', 'Unknown\n(bleaching)', 'Low', 'Mid', 'High'])
        counter += 1
        for bplot in box_plot:
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
plt.suptitle("Parameter Median Value and Distribution per Statistic Grouped by Severity", size=16, y=0.92)
plt.savefig('figures/Box and Whisker Grouped by Severity - Window 10.png', dpi=300, bbox_inches='tight')
plt.show()

# Compute tSNE
mean_values = []
min_values = []
max_values = []
var_values = []
for variable in variables:
    mean_values.append(db['mean_' + variable])
    max_values.append(db['max_' + variable])
    min_values.append(db['min_' + variable])
    var_values.append(db['variance_' + variable])
mean_values = np.array(mean_values)
min_values = np.array(min_values)
max_values = np.array(max_values)
var_values = np.array(var_values)

values = np.concatenate([mean_values, min_values, max_values, var_values])
filtered_values = []
severities = []
for row in range(values.shape[1]):
    if sum(np.isnan(values[:,row])) == 0:
        filtered_values.append(values[:,row])
        severities.append(db['severity'][row])

filtered_values = np.array(filtered_values)
severities = np.array(severities)
#filtered_values[np.isnan(filtered_values)] = 0

x_tsne = TSNE(learning_rate=100).fit_transform(filtered_values)
x_tsne = np.array(x_tsne)

colors =['#2B2D42', '#8D99AE', '#D65959', '#C93434', '#AC0808']
ordered_severity = [0, -1, 1, 2, 3]
counter = 0
plt.figure(figsize=(12, 12))
for severity in ordered_severity:
    index = severities == severity
    plt.scatter(x_tsne[index,0], x_tsne[index,1], s=1, color=colors[counter])
    counter += 1
plt.legend(ordered_severity)
plt.xticks([])
plt.yticks([])
plt.title('2D tSNE Projection of 24d Vector of Mean Values for all Statistics and Parameters')
plt.savefig('figures/tSNE Mean, Min, Max, Var - Window 10 - Params 6 - No NaNs.png', dpi=300, bbox_inches='tight')
plt.show()
