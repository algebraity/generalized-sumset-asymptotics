from ookami import *
import csv
import os
from multiprocessing import Pool, cpu_count


os.makedirs("data", exist_ok=True)
max_d = 3
max_r = 5
max_n = 80


def chunk_values(values, num_chunks):
    if num_chunks <= 0:
        return [values]
    chunk_size = max(1, (len(values) + num_chunks - 1) // num_chunks)
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def compute_chunk_rows(n_chunk):
    rows = []
    d_values = range(1, max_d)
    r_values = range(2, max_r)
    for n in n_chunk:
        aps_for_n = {d: rand_ap(1, d, n) for d in d_values}
        gps_for_n = {r: rand_gp(1, r, n) for r in r_values}

        for d in d_values:
            for r in r_values:
                a_set = aps_for_n[d] * gps_for_n[r]
                rows.append([d, r, n, a_set.cardinality, a_set.ads_cardinality])
        print(f"Process {os.getpid()} finished n={n}", flush=True)
    return rows


def main():
    n_values = list(range(5, max_n))
    num_cpus = 20
    n_chunks = chunk_values(n_values, num_cpus)

    header = ["d", "r", "n", "|A|", "|A+A|"]
    with open("data/sets_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        total_rows = 0

        with Pool(processes=num_cpus) as pool:
            for rows in pool.imap(compute_chunk_rows, n_chunks):
                writer.writerows(rows)
                total_rows += len(rows)
                file.flush()
                rows.clear()
                del rows

    print(f"Wrote {total_rows} rows to data/sets_data.csv using {num_cpus} processes")


if __name__ == "__main__":
    main()
