#!/home/vorochta/miniconda3/envs/tfenv/bin/python
import csv
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader



class OnlineAugmentationDataset(Dataset):
    """
    Vylepšený dataset s online augmentací načítající signály z CSV souboru.

    """

    def __init__(self, csv_file, segment_length=40000, positive_per_class=2,
                 negative_per_batch=12, iterations_per_epoch=100, num_classes=11,
                 min_negative_padding_ratio=0.5, batch_size=32):
        super().__init__()
        self.csv_file = csv_file
        self.segment_length = segment_length
        self.positive_per_class = positive_per_class
        self.negative_per_batch = negative_per_batch
        self.iterations_per_epoch = iterations_per_epoch
        self.num_classes = num_classes
        self.num_positive_classes = num_classes - 1  # 10 pozitivních tříd (1-10)
        self.min_negative_padding_ratio = min_negative_padding_ratio
        self.batch_size = batch_size

        # Načtení a organizace dat podle labelů
        self.data_by_label = self._load_and_organize_data()

        # Seznam všech signálů pro efektivní náhodný výběr
        self.all_signals = []
        for label in range(1, self.num_positive_classes + 1):
            self.all_signals.extend(self.data_by_label[label])

        # Kontrola dostupnosti dat
        self._validate_data_availability()

    def _load_and_organize_data(self):
        """Načte CSV a organizuje data podle labelů."""
        data_by_label = {label: [] for label in range(1, self.num_positive_classes + 1)}

        print(f"Načítání dat z {self.csv_file}...")
        with open(self.csv_file, newline='') as infile:
            csv_reader = csv.reader(infile)
            row_count = 0
            skipped_rows = 0

            for row in csv_reader:
                row_count += 1
                try:
                    row_float = np.array(row, dtype=float)
                except ValueError:
                    skipped_rows += 1
                    continue

                if len(row_float) < 4:
                    skipped_rows += 1
                    continue

                d_start = int(row_float[-3])
                d_end = int(row_float[-2])
                signal_label = int(row_float[-1])
                raw_signal = row_float[:-3]

                # Kontrola validity dat
                if signal_label < 1 or signal_label > self.num_positive_classes:
                    skipped_rows += 1
                    continue

                if d_start >= d_end or d_start < 0 or d_end >= len(raw_signal):
                    skipped_rows += 1
                    continue

                signal_data = {
                    'raw_signal': raw_signal,
                    'd_start': d_start,
                    'd_end': d_end,
                    'label': signal_label
                }

                data_by_label[signal_label].append(signal_data)

        print(f"Načteno {row_count} řádků, přeskočeno {skipped_rows} řádků")
        for label, signals in data_by_label.items():
            print(f"Label {label}: {len(signals)} signálů")

        return data_by_label

    def _validate_data_availability(self):
        """Kontroluje, zda je k dispozici dostatek dat pro každý label."""
        for label in range(1, self.num_positive_classes + 1):
            if len(self.data_by_label[label]) < self.positive_per_class:
                raise ValueError(f"Nedostatek dat pro label {label}: "
                                 f"potřeba {self.positive_per_class}, k dispozici {len(self.data_by_label[label])}")

    def _normalize_signal(self, signal):
        """Normalizuje signál."""
        signal = np.array(signal, dtype=np.float32)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val != 0:
            normalized_signal = (signal - mean_val) / std_val
        else:
            normalized_signal = signal - mean_val
        return normalized_signal

    def _extract_positive_segment(self, signal_data):
        """Vytvoří pozitivní segment obsahující gen."""
        raw_signal = signal_data['raw_signal']
        d_start = signal_data['d_start']
        d_end = signal_data['d_end']

        gene_length = d_end - d_start
        signal_length = len(raw_signal)

        if gene_length >= self.segment_length:
            # Gen je delší než segment - vybereme náhodnou část genu
            max_start_in_gene = d_end - self.segment_length
            segment_start = random.randint(d_start, max_start_in_gene)
            segment_end = segment_start + self.segment_length
        else:
            # Gen je kratší než segment - segment bude obsahovat celý gen
            # Výjimečný případ - celý signál je kratší než segment
            if signal_length < self.segment_length:
                segment = raw_signal.copy()
                # Doplníme nulami do požadované délky
                padding_needed = self.segment_length - signal_length
                segment = np.pad(segment, (0, padding_needed), 'constant', constant_values=0)
                return segment

            # Základní interval pro segment_start: [d_end - segment_length, d_start]
            left_basic = d_end - self.segment_length
            right_basic = d_start

            # Omezení zleva (začátek signálu): segment_start >= 0
            left_boundary = max(0, left_basic)

            # Omezení zprava (konec signálu): segment_start <= signal_length - segment_length
            right_boundary = min(right_basic, signal_length - self.segment_length)

            # Kontrola zda máme platný interval
            if left_boundary > right_boundary:
                return None

            # Náhodný výběr segment_start z platného intervalu
            segment_start = random.randint(left_boundary, right_boundary)
            segment_end = segment_start + self.segment_length

        segment = raw_signal[segment_start:segment_end]

        if len(segment) != self.segment_length:
            return None

        return segment

    def _extract_negative_segments_from_signal(self, signal_data):
        """
        Vytvoří negativní segmenty z jednoho signálu.
        """
        raw_signal = signal_data['raw_signal']
        d_start = signal_data['d_start']
        d_end = signal_data['d_end']
        signal_length = len(raw_signal)

        possible_segments = []

        # === SEGMENTY PŘED GENEM ===
        length_before_gene = d_start
        if length_before_gene >= self.segment_length:
            # Vytvoříme všechny možné nepřekrývající se segmenty před genem
            num_segments_before = length_before_gene // self.segment_length
            for i in range(num_segments_before):
                segment_start = i * self.segment_length
                segment_end = segment_start + self.segment_length
                segment = raw_signal[segment_start:segment_end]
                if len(segment) == self.segment_length:
                    possible_segments.append(segment)

        # === SEGMENTY ZA GENEM ===
        length_after_gene = signal_length - d_end
        if length_after_gene >= self.segment_length:
            # Vytvoříme všechny možné nepřekrývající se segmenty za genem
            num_segments_after = length_after_gene // self.segment_length
            for i in range(num_segments_after):
                segment_start = d_end + i * self.segment_length
                segment_end = segment_start + self.segment_length
                segment = raw_signal[segment_start:segment_end]
                if len(segment) == self.segment_length:
                    possible_segments.append(segment)
        elif length_after_gene >= int(self.segment_length * self.min_negative_padding_ratio):
            # Padding pro negativní segment za genem pokud je aspoň 50% požadované délky segmentu
            segment = raw_signal[d_end:signal_length]
            padding_needed = self.segment_length - len(segment)
            segment_padded = np.pad(segment, (0, padding_needed), 'constant', constant_values=0)
            possible_segments.append(segment_padded)

        return possible_segments

    def _get_signal_identifier(self, signal_data):
        """Vytvoří unikátní identifikátor signálu pro sledování použitých signálů (zabrání duplicitě)."""
        return (signal_data['label'], signal_data['d_start'], signal_data['d_end'])

    def _extract_all_negative_segments(self, selected_signals_data, used_signal_ids):
        """
        funkce pro extrakci všech potřebných negativních segmentů.

        1. Nejdřív extrahuje negativní segmenty z vybraných 20 signálů
        2. Pokud nestačí, náhodně vybírá další signály z už načtených dat
        """
        negative_segments = []

        # 1. FÁZE: Extrakce negativních segmentů z vybraných signálů (20 signálů)
        for signal_data in selected_signals_data:
            # Normalizace signálu
            normalized_signal = self._normalize_signal(signal_data['raw_signal'])
            signal_data_normalized = {
                'raw_signal': normalized_signal,
                'd_start': signal_data['d_start'],
                'd_end': signal_data['d_end'],
                'label': signal_data['label']
            }

            # Negativní segmenty ze signálu
            neg_segments = self._extract_negative_segments_from_signal(signal_data_normalized)
            negative_segments.extend(neg_segments)

            # Kontrola zda už máme dost
            if len(negative_segments) >= self.negative_per_batch:
                break

        # 2. FÁZE: Pokud nemáme dostatek, náhodně vybíráme další signály
        while len(negative_segments) < self.negative_per_batch:
            # Náhodný výběr signálu z už načtených dat (self.all_signals)
            signal_data = random.choice(self.all_signals)

            # Kontrola duplicity podle identifikátoru
            signal_id = self._get_signal_identifier(signal_data)
            if signal_id in used_signal_ids:
                continue  # Skip duplicates and try next signal

            # Normalizace signálu
            normalized_signal = self._normalize_signal(signal_data['raw_signal'])
            signal_data_normalized = {
                'raw_signal': normalized_signal,
                'd_start': signal_data['d_start'],
                'd_end': signal_data['d_end'],
                'label': signal_data['label']
            }

            # Pokus o vytvoření negativních segmentů
            neg_segments = self._extract_negative_segments_from_signal(signal_data_normalized)
            negative_segments.extend(neg_segments)

            # Označíme signál jako použitý
            used_signal_ids.add(signal_id)

            # Kontrola zda už máme dost (s rezervou)
            if len(negative_segments) >= self.negative_per_batch:
                break

        # 3. FÁZE: Náhodný výběr přesně tolika segmentů kolik potřebujeme
        if len(negative_segments) > self.negative_per_batch:
            negative_segments = random.sample(negative_segments, self.negative_per_batch)

        return negative_segments

    def __len__(self):
        return self.iterations_per_epoch

    def __getitem__(self, idx):
        """Vytvoří jeden batch s pozitivními a negativními segmenty."""
        # Sledování použitých signálů v tomto batchi
        used_signal_ids = set()

        # Vybíráme signály pro pozitivní segmenty
        selected_signals = []
        selected_signals_data = []  # Pro negativní segmenty

        random_labels_pos = [random.randint(1, self.num_positive_classes) for _ in range(random.randint(1, self.batch_size-1))]
        self.negative_per_batch  = self.batch_size - len(random_labels_pos)
        for label in random_labels_pos:
        # for label in range(1, self.num_positive_classes + 1):
            available_signals = self.data_by_label[label]
            selected = random.sample(available_signals, self.positive_per_class)

            # Označíme tyto signály jako použité
            for signal_data in selected:
                signal_id = self._get_signal_identifier(signal_data)
                used_signal_ids.add(signal_id)
                selected_signals_data.append(signal_data)

            selected_signals.extend([(signal_data, label) for signal_data in selected])

        # Vytváříme pozitivní segmenty
        positive_segments = []

        for signal_data, label in selected_signals:
            # Normalizace signálu
            normalized_signal = self._normalize_signal(signal_data['raw_signal'])
            signal_data_normalized = {
                'raw_signal': normalized_signal,
                'd_start': signal_data['d_start'],
                'd_end': signal_data['d_end'],
                'label': signal_data['label']
            }

            # Pozitivní segment
            pos_segment = self._extract_positive_segment(signal_data_normalized)
            if pos_segment is not None:
                positive_segments.append((pos_segment, label))

        # extrakce všech negativních segmentů
        negative_segments_raw = self._extract_all_negative_segments(selected_signals_data, used_signal_ids)
        negative_segments = [(segment, 0) for segment in negative_segments_raw]

        # Kontrola - pokud stále nemáme dostatek, vyhodíme chybu
        if len(negative_segments) < self.negative_per_batch:
            raise RuntimeError(f"Nedostatek negativních segmentů: potřeba {self.negative_per_batch}, "
                               f"k dispozici {len(negative_segments)}. Zkuste snížit segment_length nebo negative_per_batch.")

        # Vytvoření finálního batche
        batch_segments = []
        batch_labels = []

        # Přidáme všechny pozitivní segmenty
        for segment, label in positive_segments:
            batch_segments.append(segment)
            batch_labels.append(label)

        # Přidáme všechny negativní segmenty
        for segment, label in negative_segments:
            batch_segments.append(segment)
            batch_labels.append(label)

        # Shufflování pořadí segmentů v batchi
        combined = list(zip(batch_segments, batch_labels))
        random.shuffle(combined)
        batch_segments, batch_labels = zip(*combined)

        # Převod na torch tensory
        batch_segments = torch.tensor(np.array(batch_segments), dtype=torch.float32)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return batch_segments, batch_labels


class SimpleValidationDataset(Dataset):
    """
    Jednoduchý validační dataset načítající předem nařezané segmenty.
    Očekávaný formát CSV: segment_data_1, segment_data_2, ..., segment_data_N, label
    Bez online augmentace, pouze načte segmenty a pošle je do modelu.
    """

    def __init__(self, csv_file, segment_length=40000):
        super().__init__()
        self.csv_file = csv_file
        self.segment_length = segment_length

        # Načtení všech segmentů
        self.segments, self.labels = self._load_preprocessed_segments()

        print(f"Validační dataset načten: {len(self.segments)} segmentů")

        # Statistiky labelů
        from collections import Counter
        label_counts = Counter(self.labels)
        print(f"Rozložení labelů ve validaci: {dict(label_counts)}")

    def _load_preprocessed_segments(self):
        """Načte předem nařezané segmenty z CSV."""
        segments = []
        labels = []

        print(f"Načítání předem nařezaných segmentů z {self.csv_file}...")

        with open(self.csv_file, newline='') as infile:
            csv_reader = csv.reader(infile)
            row_count = 0
            skipped_rows = 0

            for row in csv_reader:
                row_count += 1

                try:
                    row_float = np.array(row, dtype=float)
                except ValueError:
                    skipped_rows += 1
                    continue

                if len(row_float) < 2:  # Minimálně segment + label
                    skipped_rows += 1
                    continue

                # Posledná hodnota je label, zbytek je segment
                label = int(row_float[-1])
                segment_data = row_float[:-1]

                # Normalizace segmentu
                normalized_segment = self._normalize_segment(segment_data)

                segments.append(normalized_segment)
                labels.append(label)

        print(f"Načteno {row_count} řádků, přeskočeno {skipped_rows} řádků")

        return segments, labels

    def _normalize_segment(self, segment):
        """Normalizuje segmen. podle zadaného vzorce."""
        segment = np.array(segment, dtype=np.float32)
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        if std_val != 0:
            normalized_segment = (segment - mean_val) / std_val
        else:
            normalized_segment = segment - mean_val
        return normalized_segment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """Vrátí jeden segment a jeho label."""
        segment = torch.tensor(self.segments[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return segment, label


def get_online_dataloader(csv_file, batch_size=32, segment_length=40000,
                          positive_per_class=2, negative_per_batch=12,
                          iterations_per_epoch=100, num_classes=11,
                          shuffle=True, num_workers=0, validation=False,
                          min_negative_padding_ratio=0.5):
    """
    Vytvoří DataLoader s online augmentací nebo jednoduchý validační dataloader.

    Args:
        csv_file: cesta k CSV souboru
        batch_size: velikost batche
        segment_length: délka segmentů
        positive_per_class: počet pozitivních segmentů na třídu (ignorováno pro validaci)
        negative_per_batch: počet negativních segmentů na batch (ignorováno pro validaci)
        iterations_per_epoch: počet iterací na epochu (ignorováno pro validaci)
        num_classes: celkový počet tříd (včetně negativní)
        shuffle: zda míchat
        num_workers: počet worker procesů
        validation: zda se jedná o validační dataset (jiný formát CSV)
        min_negative_padding_ratio: minimální poměr délky pro padding negativních segmentů
    """
    if validation:
        # Jednoduchý validační dataset pro předem nařezané segmenty
        dataset = SimpleValidationDataset(csv_file, segment_length)
        # Normální DataLoader s možností shufflování a standardní batch size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        # Trénovací dataset s online augmentací
        dataset = OnlineAugmentationDataset(csv_file, segment_length, positive_per_class,
                                            negative_per_batch, iterations_per_epoch, num_classes,
                                            min_negative_padding_ratio, batch_size)
        # Batch size = 1 protože každý __getitem__ už vrací celý batch
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return loader


# Příklad použití:
if __name__ == "__main__":
    # Konfigurace
    train_csv = "/home/vorochta/verze_testu_pro_server/TimeSeriesProject-main/train_10G_balanced.csv"
    val_csv = "/home/vorochta/verze_testu_pro_server/TimeSeriesProject-main/valid_segments.csv"  # Předem nařezané segmenty
    segment_length = 40000
    positive_per_class = 2  # 2 pozitivní na třídu
    num_positive_classes = 10  # třídy 1-10
    negative_per_batch = 12  # 12 negativních na batch
    num_classes = 11  # 10 pozitivních + 1 negativní
    iterations_per_epoch = 100

    # Očekávaná velikost batche pro trénování
    expected_train_batch_size = positive_per_class * num_positive_classes + negative_per_batch  # 2*10 + 12 = 32

    print("=== TEST OPTIMALIZOVANÉHO DATALOADERU ===")

    # Trénovací dataloader (online augmentace)
    train_loader = get_online_dataloader(
        csv_file=train_csv,
        batch_size=expected_train_batch_size,
        segment_length=segment_length,
        positive_per_class=positive_per_class,
        negative_per_batch=negative_per_batch,
        iterations_per_epoch=iterations_per_epoch,
        num_classes=num_classes,
        validation=False
    )

    # Validační dataloader (předem nařezané segmenty)
    val_loader = get_online_dataloader(
        csv_file=val_csv,
        batch_size=32,  # Normální batch size pro validaci
        segment_length=segment_length,
        shuffle=True,  # Můžeme shufflovat validační data
        validation=True
    )

    print("\n=== TEST TRÉNOVACÍHO DATALOADERU ===")
    # Test několika trénovacích batchů
    for batch_idx, batch_data in enumerate(train_loader):
        batch_segments, batch_labels = batch_data
        # batch_segments shape: [1, 32, 40000] (kvůli batch_size=1 v DataLoader)
        # batch_labels shape: [1, 32]

        # Rozbalení pro kompatibilitu s trénovacím skriptem
        signals = batch_segments.squeeze(0)  # [32, 40000]
        labels = batch_labels.squeeze(0)  # [32]

        print(f"Train Batch {batch_idx + 1}: signals {signals.shape}, labels {labels.shape}")
        print(f"Unique labels: {torch.unique(labels).tolist()}")

        # Statistiky labelů
        from collections import Counter

        label_counts = Counter(labels.tolist())
        print(f"Label counts: {dict(label_counts)}")

        # Kontrola pořadí labelů (zda jsou shufflované)
        first_10_labels = labels[:10].tolist()
        print(f"Prvních 10 labelů: {first_10_labels}")

        if batch_idx >= 1:  # Test pouze 2 batche
            break

    print("\n=== TEST VALIDAČNÍHO DATALOADERU ===")
    # Test validačního dataloaderu
    for batch_idx, batch_data in enumerate(val_loader):
        segments, labels = batch_data
        print(f"Val Batch {batch_idx + 1}: segments {segments.shape}, labels {labels.shape}")
        print(f"Unique labels: {torch.unique(labels).tolist()}")

        # Statistiky labelů
        from collections import Counter

        label_counts = Counter(labels.tolist())
        print(f"Label counts: {dict(label_counts)}")

        if batch_idx >= 1:  # Test pouze 2 batche
            break

    print("\n✓ Test dokončen úspěšně!")