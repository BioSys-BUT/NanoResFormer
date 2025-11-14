import numpy as np
import csv
import matplotlib.pyplot as plt

# csv_path = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/mixed_test_5%_10G_unsegmented.csv"
csv_path = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data\mixed_test_10G_fixed_multilabel_with_ID.csv"


gene_lengths = []
No_gene_lengths = []

with open(csv_path, 'r') as infile:
    csv_reader = csv.reader(infile)
    for row in csv_reader:
        # Find index of first '*' in row
        star_idx = row.index('*')
        row_arr = np.array(row[:star_idx], dtype=float)

        info_arr = row[star_idx+1:]

        ID = info_arr[0]
        genes_info = []
        # Každý gen má 3 čísla: start, end, label
        for i in range(1, len(info_arr), 3):
            if i + 2 < len(info_arr):
                gene_info = [info_arr[i], info_arr[i+1], info_arr[i+2]]
                genes_info.append(gene_info)
        num_genes = len(genes_info)

        # Pokud je více genů, zpracuj každý zvlášť
        signal_labels = []
        d_starts = []
        d_ends = []
        for gene in genes_info:
            d_start = int(gene[0])
            d_end = int(gene[1])
            signal_label = int(gene[2])

            d_starts.append(d_start)
            d_ends.append(d_end)
            signal_labels.append(signal_label)

        if all(label == 0 for label in signal_labels):
            raw_signal = row_arr[:-3]
            No_gene_lengths.append(len(raw_signal))
        else:
            raw_signal = row_arr[:-3]
            gene_lengths.append(len(raw_signal))
            

# plt.hist([gene_lengths, No_gene_lengths], bins=100, label=['Geny', 'No Geny'], alpha=0.7)
# plt.xlabel('Délka signálu')
# plt.ylabel('Počet')
# plt.title('Histogram délek signálu pro obě kategorie')
# plt.legend()
# plt.show()

all_lengths = gene_lengths + No_gene_lengths
counts, bins, patches = plt.hist(all_lengths, bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('Signal length')
plt.ylabel('Count')
# plt.title('Histogram of signal lengths (Genes + No Genes)')

# Nastav celočíselné popisky na ose x po 5000
xticks = [int(bins[0])] + np.arange(0, int(bins[-1]), 50000).tolist()
plt.xticks(xticks)

# Ujisti se, že první sloupec má číselný popisek
plt.gca().set_xticks(list(plt.gca().get_xticks()) + [bins[0]])
plt.gca().set_xticklabels([str(int(x)) for x in plt.gca().get_xticks()])

plt.show()
