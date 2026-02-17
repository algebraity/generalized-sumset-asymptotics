from ookami import *
import csv
import os
from multiprocessing import Pool, cpu_count


os.makedirs("data", exist_ok=True)
max_r = 10
max_n = 40


def chunk_values(values, num_chunks):
    if num_chunks <= 0:
        return [values]
    chunk_size = max(1, (len(values) + num_chunks - 1) // num_chunks)
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def compute_chunk_rows(n_chunk):
    rows = []
    r_values = range(2, max_r + 1)
    d_value = 1

    for n in n_chunk:
        ap_set = rand_ap(1, d_value, n)
        gps_for_n = {r: rand_gp(1, r, n) for r in r_values}

        for r in r_values:
            a_set = ap_set * gps_for_n[r]
            rows.append([r, n, a_set.cardinality, a_set.ads_cardinality])

        print(f"Process {os.getpid()} finished n={n}", flush=True)

    return rows


def main():
    n_values = list(range(5, max_n + 1))
    num_cpus = cpu_count()
    n_chunks = chunk_values(n_values, num_cpus)

    header = ["r", "n", "|A|", "|A+A|"]
    with open("data/Anr_data.csv", "w", newline="") as file:
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

    print(f"Wrote {total_rows} rows to data/Anr_data.csv using {num_cpus} processes")


if __name__ == "__main__":
    main()