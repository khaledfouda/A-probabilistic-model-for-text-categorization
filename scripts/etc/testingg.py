
if __name__ == '__main__':
    import pandas as pd
    from clean_gen import clean
    from tabulate import tabulate
    from prototypical_fit import ProtoFit

    d = pd.read_csv("../data/etc/Political Social Media Posts/political_social_media.csv", encoding="ISO-8859-1")
    data = clean(d, 'text', 'bias', 'testDELETELATER')
    print(tabulate(data.sample(1), headers='keys', tablefmt='psql'))

    ob = ProtoFit('KAGGLE', k=300, log_to_file=True)
    _ = ob.fit(data, n_drops=10, valid_size=.1)
