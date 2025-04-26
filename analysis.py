import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from itertools import combinations
from Bio import SeqIO
from Bio import Align
import io
from statsmodels.stats.multitest import multipletests


sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Load data
data = pd.read_csv("./data/IGHV1-2_usages.csv")
print(f"Number of individuals: {len(data)}")

# 2. 
print("Problem 2: # individuls and Mean usage for each haplotype")
haplotype_stats = {}
unique_haplotypes = data['Haplotype'].unique()
#print(unique_haplotypes)
for haplotype in unique_haplotypes:
    haplotype_data = data[data['Haplotype'] == haplotype]
    n_individuals = len(haplotype_data)
    mean_usage = haplotype_data['Usage'].mean()
    haplotype_stats[haplotype] = {
        '# individuals': n_individuals,
        'Mean usage': mean_usage
    }

# Create a DataFrame for Table 1
table1 = pd.DataFrame.from_dict(haplotype_stats, orient='index')
table1.index.name = 'Haplotype'
table1 = table1.reset_index()
print("Table 1:")
print(table1)

# 3. ANOVA + Bonferroni
print("Problem 3: ANOVA test for each pair of haplotypes")
haplotype_pairs = list(combinations(unique_haplotypes, 2))
print(haplotype_pairs)
p_values = {}

for h1, h2 in haplotype_pairs:
    usages1 = data[data['Haplotype'] == h1]['Usage']
    usages2 = data[data['Haplotype'] == h2]['Usage']
    _, p_val = f_oneway(usages1, usages2)
    p_values[(h1, h2)] = p_val

# Bonferroni correction
alpha = 0.05
n_tests = len(p_values)
bonferroni_threshold = alpha / n_tests
print("Bonferroni threshold: ",bonferroni_threshold)
# Create Table 2 (p-values table)
table2 = pd.DataFrame(index=unique_haplotypes, columns=unique_haplotypes)
for i in unique_haplotypes:
    for j in unique_haplotypes:
        if i == j:
            table2.loc[i, j] = "-"
        elif (i, j) in p_values:
            p_val = p_values[(i, j)]
            if p_val < bonferroni_threshold:
                table2.loc[i, j] = f"{p_val:.4e}*"
            else:
                table2.loc[i, j] = f"{p_val:.4e}"
        else:  # (j, i) must be in p_values
            p_val = p_values[(j, i)]
            if p_val < bonferroni_threshold:
                table2.loc[i, j] = f"{p_val:.4e}*"
            else:
                table2.loc[i, j] = f"{p_val:.4e}"

print("\nTable 2 (p-values matrix):")
print(table2)

# Create boxplot of usages across haplotypes
plt.figure(figsize=(12, 8))
sns.boxplot(x='Haplotype', y='Usage', data=data)
plt.title('IGHV1-2 Usage by Haplotype')
plt.xlabel('Haplotype')
plt.ylabel('Usage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('haplotype_usages_boxplot.png')
plt.close()

# 4. 
print("Problem 4: MSA and SNP")

def extract_alleles_from_haplotypes(unique_haplotypes):
    all_alleles = set()
    for haplotype in unique_haplotypes:
        if "-" in str(haplotype):
            alleles = str(haplotype).split("-")
            for allele in alleles:
                all_alleles.add(f"IGHV1-2*0{allele}")
        else:
            all_alleles.add(f"IGHV1-2*0{haplotype}")
    return all_alleles

all_alleles = extract_alleles_from_haplotypes(unique_haplotypes)
print(f"Found {len(all_alleles)} unique alleles to analyze: {', '.join(all_alleles)}")


def read_sequences_from_fasta(fasta_file, target_alleles):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id in target_alleles:
            sequences[record.id] = str(record.seq)
    return sequences


allele_sequences = read_sequences_from_fasta("./data/IGHV.fa", all_alleles)

import subprocess
import os

def save_sequences_to_fasta(sequences, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")

extracted_fasta_path = "./data/extracted_alleles.fasta"
save_sequences_to_fasta(allele_sequences, extracted_fasta_path)

def align_with_mafft_command_line(input_fasta, output_fasta):

    print(f"Running MAFFT alignment on sequences in {input_fasta}...")
    cmd = ['mafft', '--auto', input_fasta]
    with open(output_fasta, 'w') as out_file:
        result = subprocess.run(cmd, stdout=out_file, stderr=subprocess.PIPE, text=True)

    aligned_sequences = {}
    for record in SeqIO.parse(output_fasta, "fasta"):
        aligned_sequences[record.id] = str(record.seq)
    
    return aligned_sequences


aligned_fasta_path = "./data/aligned_alleles.fasta"

aligned_sequences = align_with_mafft_command_line(extracted_fasta_path, aligned_fasta_path)

def identify_snps(aligned_sequences):
    seq_list = list(aligned_sequences.values())
    allele_names = list(aligned_sequences.keys())
    
    snps = []
    if not seq_list:
        return [], pd.DataFrame()
    
    seq_length = len(seq_list[0])
    for pos in range(seq_length):
        nucleotides = set(seq[pos].upper() for seq in seq_list if pos < len(seq))

        nucleotides.discard('-')
        
        if len(nucleotides) > 1:
            snps.append(pos)
    

    table3_data = []
    for i, allele in enumerate(allele_names):
        snp_pairs = []
        for pos in snps:
            if pos < len(seq_list[i]) and seq_list[i][pos] != '-':
                snp_pairs.append(f"({seq_list[i][pos].upper()}, {pos+1})")
        
        table3_data.append({
            'Allele': allele,
            'SNP Pairs': ", ".join(snp_pairs)
        })
    
    table3 = pd.DataFrame(table3_data)
    
    return snps, table3


snp_positions, table3 = identify_snps(aligned_sequences)

print("Table 3 SNP:")
print(table3)

# 5. 
print("Problem 5: SNP states for each haplotype")

def compute_snp_states(unique_haplotypes, snp_positions, aligned_sequences):
    haplotype_snp_states = {}
    
    for haplotype in unique_haplotypes:
        # hetero
        if "-" in str(haplotype):
            allele_nums = str(haplotype).split("-")
            allele_ids = [f"IGHV1-2*0{num}" for num in allele_nums]
        else:
            # homo
            allele_ids = [f"IGHV1-2*0{haplotype}"]
        
        states = []
        for pos in snp_positions:
            nucleotides = set()
            for allele_id in allele_ids:
                if allele_id in aligned_sequences and pos < len(aligned_sequences[allele_id]):
                    nucleotide = aligned_sequences[allele_id][pos].upper()
                    if nucleotide != '-': 
                        nucleotides.add(nucleotide)
            
            if len(nucleotides) == 1:  # homo pos
                states.append(nucleotides.pop())
            elif len(nucleotides) > 1:  # hetero pos
                states.append("/".join(sorted(nucleotides)))
            else: 
                states.append("N/A")
        
        haplotype_snp_states[haplotype] = states

    table4_data = []
    for haplotype, states in haplotype_snp_states.items():
        table4_data.append({
            'Haplotype': haplotype,
            'SNP States': ", ".join(states)
        })
    
    return pd.DataFrame(table4_data)


table4 = compute_snp_states(unique_haplotypes, snp_positions, aligned_sequences)

print("Table 4 SNP states for each haplotype:")
print(table4)

print("Table 4 SNP states for each haplotype:")
print(table4)

# 6. 
print("Problem 6: Association between SNP states and usages")

def analyze_snp_associations(data, table4, snp_positions):
    haplotype_to_snp_states = {}
    for _, row in table4.iterrows():
        haplotype = row['Haplotype']
        states = row['SNP States'].split(', ')
        haplotype_to_snp_states[haplotype] = states
    data_copy = data.copy()
    snp_p_values = {}
    
    for i, pos in enumerate(snp_positions):
        snp_name = f"SNP_{pos+1}"
        
        data_copy[snp_name] = data_copy['Haplotype'].apply(
            lambda h: haplotype_to_snp_states.get(h, ['N/A'])[i] if i < len(haplotype_to_snp_states.get(h, [])) else 'N/A'
        )
        
        groups = []
        states = data_copy[snp_name].unique()
        state_groups = {}
        
        for state in states:
                
            group = data_copy[data_copy[snp_name] == state]['Usage']
            if len(group) > 0:
                groups.append(group)
                state_groups[state] = group.values
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            snp_p_values[snp_name] = p_val
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=snp_name, y='Usage', data=data_copy)
            plt.title(f'IGHV1-2 Usage by {snp_name} State (p={p_val:.4e})')
            plt.xlabel(f'{snp_name} State')
            plt.ylabel('Usage')
            plt.tight_layout()
            plt.savefig(f'{snp_name}_usages_boxplot.png')
            plt.close()
            
            print(f"\nStatistics for {snp_name}:")
            for state, values in state_groups.items():
                print(f"  State {state}: n={len(values)}, mean={np.mean(values):.5f}, std={np.std(values):.5f}")
    
    alpha = 0.05
    n_tests = len(snp_p_values)
    if n_tests > 0:
        bonferroni_threshold = alpha
        
        print("\nSNP Association Results :")
        for snp, p_val in snp_p_values.items():
            significance = "Significant" if p_val < bonferroni_threshold else "Not significant"
            print(f"{snp}: p-value = {p_val:.4e} ({significance} at alpha {alpha}")
        
        significant_snps = [snp for snp, p_val in snp_p_values.items() if p_val < bonferroni_threshold]

    
    return snp_p_values


snp_associations = analyze_snp_associations(data, table4, snp_positions)