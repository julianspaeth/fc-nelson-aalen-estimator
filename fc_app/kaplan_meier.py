import itertools
import numpy as np
import pandas as pd
from flask import current_app
from scipy import stats
from statsmodels.stats.multitest import multipletests

from fc_app.plots import plot_km
from redis_util import redis_get, redis_set


def read_input(input_dir: str):
    """
    Reads all files stored in 'input_dir'.
    :param input_dir: The input directory containing the files.
    :return: None
    """
    data = None
    current_app.logger.info('[API] Parsing data of ' + input_dir)
    columns = [redis_get('duration_col'), redis_get('event_col')]
    duration_col = redis_get('duration_col')
    event_col = redis_get('event_col')
    if redis_get('category_col') is not None:
        columns.append(redis_get('category_col'))
    current_app.logger.info("Columns: " + str(columns))
    filename = redis_get('input_filename')
    try:
        if filename.endswith(".csv"):
            sep = ','
            data = pd.read_csv(input_dir + '/' + filename, sep=sep, usecols=columns).sort_values(
                by=duration_col)
        elif filename.endswith(".tsv"):
            sep = '\t'
            data = pd.read_csv(input_dir + '/' + filename, sep=sep, usecols=columns).sort_values(
                by=duration_col)
        elif filename.endswith(".sas7bdat"):
            data = pd.read_sas(input_dir + '/' + filename, encoding='iso-8859-1').sort_values(by=duration_col)
            data = data.loc[:, columns]
        data[[duration_col, event_col]] = data[[duration_col, event_col]].astype("float")
        data = data.rename({duration_col: "time", event_col: "event_observed"}, axis=1)
        data = data.dropna()

        current_app.logger.info(str(data.shape[0]) + " samples")
    except Exception as e:
        current_app.logger.info("Error in reading file: " + str(e))
        quit()

    return data


def calculate_global():
    """
    Calculates the global KM of the data of all clients.
    :return: None
    """
    current_app.logger.info('[API] Calculate Global KM')
    client_data = redis_get('global_data')
    results, p_values = compute_global_km(client_data, redis_get('differential_privacy'))
    current_app.logger.info(f'[API] Global Results calculated')
    redis_set('global_km', [results, p_values])
    current_app.logger.info(results)
    current_app.logger.info(p_values)
    current_app.logger.info(redis_get('global_km'))


def write_results(results, output_dir: str):
    """
    Writes the results of global_km to the output_directory.
    :param results: Global results calculated from the local counts of the clients
    :param output_dir: String of the output directory. Usually /mnt/output
    :return: None
    """
    current_app.logger.info("[API] Write results to output folder")
    result = pd.DataFrame().from_dict(results[0], dtype=float).drop_duplicates()
    result.index = result.index.astype(float)
    result = result.sort_index()

    if redis_get('category_col') is not None:
        p_values = pd.DataFrame().from_dict(results[1], dtype=float)
        try:
            p_values.to_csv(output_dir + '/' + redis_get('logrank_test_filename') + '.csv')
        except Exception as e:
            current_app.logger.error("Error while writing the logrank test to file")
            current_app.logger.error(str(e))
    try:
        result.to_csv(output_dir + '/' + redis_get('survival_function_filename') + '.csv')
    except Exception as e:
        current_app.logger.error("Error while writing survival function to file")
        current_app.logger.error(str(e))
    try:
        surv_plot = plot_km(data=result)
        surv_plot.savefig(output_dir + '/' + redis_get('survival_plot_filename') + '.png', format='png')
    except Exception as e:
        current_app.logger.error("Error while creating plots")
        current_app.logger.error(str(e))


def calculate_local_counts():
    """
    Calculate the local counts of a client
    :return: the local counts and the number of samples
    """
    current_app.logger.info('[API] Calculate local counts')
    data = redis_get('files')

    if data is None:
        current_app.logger.info('[API] No data available')
        return None
    else:
        local_results = {}
        category_col = redis_get('category_col')
        if category_col:
            for category in data[category_col].unique():
                current_app.logger.info("Compute counts for " + str(category))
                df_col = data.loc[data[category_col] == category]
                d_n_matrix = compute_d_and_n_matrix(df_col)
                local_results[str(category)] = d_n_matrix.astype(float).to_dict()
        else:
            current_app.logger.info("No categories found. Compute counts for complete dataset")
            d_n_matrix = compute_d_and_n_matrix(data)
            local_results["complete"] = d_n_matrix.astype(float).to_dict()

        return local_results


def compute_d_and_n(data, t):
    temp = data[data["time"] == t].groupby("event_observed").count()
    try:
        d = temp.loc[1.0, "time"]
    except KeyError:
        d = 0
    try:
        n = temp.loc[0.0, "time"] + d
    except KeyError:
        n = 0 + d
    return d, n


def compute_d_and_n_matrix(data):
    timeline = data["time"].sort_values().unique()
    di = np.full(len(timeline) + 1, np.nan)
    ni = np.full(len(timeline) + 1, np.nan)
    ni[0] = data.shape[0]
    for i in range(len(timeline)):
        d, n = compute_d_and_n(data, timeline[i])
        di[i] = d
        ni[i + 1] = ni[i] - n
    m = pd.DataFrame(index=timeline)
    m["di"] = di[:-1]
    m["ni"] = ni[:-1]

    return m


def compute_global_km(client_data, privacy_level: str = 'none'):
    time = {}
    data = {}
    results = {"KM": None, "NA": None}
    for client_result in client_data:
        for category in client_result.keys():
            client_result[category].index = client_result[category].index.astype(float)
            if category not in data.keys():
                data[category] = []
                time[category] = []
            data[category].append(client_result[category])
            time[category] += list(client_result[category].index)
    timelines = dict((k, sorted(set(v))) for k, v in time.items())
    aggregation_dfs = {}
    for category in data.keys():
        aggregation_df = pd.DataFrame(index=timelines[category], columns=["di", "ni"])
        for t in timelines[category]:
            di = 0
            ni = 0
            for c in data[category]:
                try:
                    di = di + float(c.loc[t, "di"])
                except KeyError:
                    pass
                try:
                    ni = ni + float(c.loc[t, "ni"])
                except KeyError:
                    indices = list(c.index)
                    try:
                        res = list(map(lambda i: i > t, indices)).index(True)
                        ni = ni + float(c.loc[indices[res], "ni"])
                    except ValueError:
                        pass

                aggregation_df.loc[t, "di"] = di
                aggregation_df.loc[t, "ni"] = ni

        aggregation_df = aggregation_df.dropna()
        if privacy_level != "none":
            eps = None
            if privacy_level == "high":
                eps = 1
            elif privacy_level == "middle":
                eps = 2
            elif privacy_level == "low":
                eps = 3
            aggregation_df = compute_dp_matrix(eps, aggregation_df)
        aggregation_dfs[category] = aggregation_df
        surv_label = str(category)
        cum_label = str(category)
        survival_function = compute_survival_function(aggregation_df.copy(), surv_label)
        cum_hazard_rate = compute_hazard_function(aggregation_df.copy(), cum_label)
        if results["KM"] is None:
            results["KM"] = survival_function.to_frame()
        else:
            results["KM"] = pd.concat([results["KM"], survival_function], axis=1, join='outer')
        if results["NA"] is None:
            results["NA"] = cum_hazard_rate.to_frame()
        else:
            results["NA"] = pd.concat([results["NA"], cum_hazard_rate], axis=1, join='outer')

    if redis_get('category_col') is not None:
        p_values = pairwise_logrank_test(aggregation_dfs).reset_index()
        current_app.logger.info(p_values.to_dict())
        return results["KM"].to_dict(), p_values.to_dict()
    else:
        return results["KM"].to_dict(), None


def compute_dp_matrix(eps, d_n_matrix):
    # Create partial matrix M
    m = pd.DataFrame(index=d_n_matrix.index, columns=['di', 'ri'])
    # t_1 = d_1 and n_1 known
    m["di"] = d_n_matrix["di"]
    m.iloc[0, 1] = d_n_matrix.iloc[0, 1]
    # Add Laplacian Noises
    z = m.copy()
    z_di = np.random.laplace(loc=0, scale=2 / eps, size=d_n_matrix.shape[0])
    z_ri = np.random.laplace(loc=0, scale=2 / eps, size=d_n_matrix.shape[0])
    z['di'] = z_di
    z['ri'] = z_ri
    m_ = pd.DataFrame(index=d_n_matrix.index, columns=['di', 'ri'])
    m_['di'] = m['di'].astype(float) + z['di'].astype(float)
    m_['ri'] = m['ri'] + z['ri'].astype(float)

    # compute differentially number at risk
    r_last = m_.iloc[0, 1]
    d_last = m_.iloc[0, 0]
    for j in range(1, d_n_matrix.shape[0]):
        r = r_last - d_last
        d = m_.iloc[j, 0]
        if r < 0:
            m_.iloc[j, 1] = 0
        else:
            m_.iloc[j, 1] = r
        if d < 0:
            m_.iloc[j, 0] = 0
        r_last = r
        d_last = d
    # Replace negative values by 0 for di and end df if ri is 0
    m_ = m_.reset_index()
    ri_zero_index = (m_["ri"] == 0).idxmax()
    if ri_zero_index != 0:
        m_ = m_.iloc[0:ri_zero_index - 1, :]
    m_ = m_.rename({"ri": "ni"}, axis=1)
    return m_.clip(lower=0)


def compute_survival_function(m, surv_label="KM_estimate"):
    m["1-(di/ni)"] = (1 - m["di"] / m["ni"])
    m.dropna(inplace=True)
    m.reset_index(inplace=True)
    m.rename({"index": "t"}, axis=1, inplace=True)
    si = []
    for i in m.index:
        si.append(np.product(m.loc[0:i, "1-(di/ni)"].to_list()))
    m["si"] = si
    if m.iloc[0, 0] != 0.0:
        m.index = m.index + 1
        m.loc[0, :] = [0.0, 0.0, m.shape[0], 1.0, 1.0]
        m.sort_index(inplace=True)
    m.rename({"t": "timeline", "si": surv_label}, axis=1, inplace=1)
    m.set_index("timeline", inplace=True)

    return m[surv_label]


def compute_hazard_function(m, cum_label="NA_estimate"):
    m["di/ni"] = (m["di"] / m["ni"])
    m.dropna(inplace=True)
    m.reset_index(inplace=True)
    m.rename({"index": "t"}, axis=1, inplace=True)
    hi = []
    for t in m.index:
        hi.append(np.sum(m.loc[0:t, "di/ni"].to_list()))
    m["hi"] = hi
    if m.iloc[0, 0] != 0.0:
        m.index = m.index + 1
        m.loc[0, :] = [0.0, 0.0, m.shape[0], 0.0, 0.0]
        m.sort_index(inplace=True)
    m.rename({"t": "timeline", "hi": cum_label}, axis=1, inplace=1)
    m.set_index("timeline", inplace=True)

    return m[cum_label]


def logrank_test(df1, df2):
    logrank_df = pd.merge(left=df1, right=df2, left_index=True, right_index=True,
                          how='outer', suffixes=('_0', '_1'))
    logrank_df.iloc[:, [0, 2]] = logrank_df.iloc[:, [0, 2]].fillna(0)

    logrank_df = logrank_df.fillna(method='bfill')
    logrank_df = logrank_df.fillna(0)

    # compute the factors needed (from lifelines)
    N_j = logrank_df[["di_0", "di_1"]].sum(0).values
    n_ij = logrank_df[["ni_0", "ni_1"]]
    d_i = logrank_df["di_0"] + logrank_df["di_1"]
    n_i = logrank_df["ni_0"] + logrank_df["ni_1"]
    ev = n_ij.mul(d_i / n_i, axis="index").sum(0)

    # vector of observed minus expected
    Z_j = N_j - ev

    assert abs(Z_j.sum()) < 10e-8, "Sum is not zero."  # this should move to a test eventually.

    # compute covariance matrix
    factor = (((n_i - d_i) / (n_i - 1)).replace([np.inf, np.nan], 1)) * d_i / n_i ** 2
    n_ij["_"] = n_i.values
    V_ = n_ij.mul(np.sqrt(factor), axis="index").fillna(0)
    V = -np.dot(V_.T, V_)
    ix = np.arange(2)
    V[ix, ix] = V[ix, ix] - V[-1, ix]
    V = V[:-1, :-1]

    # take the first n-1 groups
    U = Z_j.iloc[:-1] @ np.linalg.pinv(V[:-1, :-1]) @ Z_j.iloc[:-1]  # Z.T*inv(V)*Z
    # compute the p-values and tests
    return stats.chi2.sf(U, 1), U


def pairwise_logrank_test(data, alpha=0.05):
    current_app.logger.info("Compute Pairwise logrank test")
    my_index = pd.MultiIndex(levels=[[], []],
                             codes=[[], []],
                             names=[u'Category 1', u'Category 2'])
    method = redis_get('multipletesting_method')
    p_method = 'p_' + str(method)
    log_p_method = 'log(p_' + str(method) + ')'
    my_columns = [u'test_statistic', u'p', u'log(p)', p_method, log_p_method]
    p_matrix = pd.DataFrame(index=my_index, columns=my_columns)
    for category, category2 in itertools.combinations(list(data.keys()), 2):
        p, U = logrank_test(data[category2], data[category])
        p_matrix.loc[(category2, category), :] = [U, p, np.nan, np.nan, np.nan]
    p_corrected = multipletests(p_matrix.loc[:, "p"].to_list(), alpha=alpha, method=method, is_sorted=False,
                                returnsorted=False)
    p_matrix[p_method] = p_corrected[1]
    p_matrix["log(p)"] = p_matrix.loc[:, "p"].apply(lambda x: -np.log2(x))
    p_matrix[log_p_method] = p_matrix.loc[:, "p"].apply(lambda x: -np.log2(x))
    p_matrix = p_matrix.astype("float32")
    p_matrix = p_matrix.reset_index()
    if p_matrix.shape[0] < 2:
        p_matrix = p_matrix.drop(p_method, axis=1)
        p_matrix = p_matrix.drop(log_p_method, axis=1)
    if "int" in str(p_matrix.dtypes["Category 1"]):
        p_matrix = p_matrix.sort_values(['Category 1', 'Category 2'], ascending=[1, 1])

    p_matrix = p_matrix.set_index(['Category 1', 'Category 2'])

    return p_matrix
