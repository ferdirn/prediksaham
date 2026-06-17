---
description: Generate and maintain comprehensive documentation from code
argument-hint: [--readme | --claude | --check]
allowed-tools: Bash(ls:*), Bash(cat:*), Bash(test:*), Bash(grep:*), Bash(find:*)

---

Perbarui dokumentasi project agar tetap sinkron dengan implementasi kode terbaru.

Dokumentasi utama project ini:
- `CLAUDE.md` — referensi lengkap untuk Claude Code: daftar file, usage CLI, konfigurasi
- `README.md` — dokumentasi publik: overview, instalasi, contoh penggunaan

## Usage

```
/doc-update              # periksa dan perbarui CLAUDE.md + README.md
/doc-update --claude     # fokus ke CLAUDE.md saja
/doc-update --readme     # fokus ke README.md saja
/doc-update --check      # hanya periksa, jangan tulis — tampilkan apa yang perlu diperbarui
```

## Alur kerja yang disarankan

Jalankan `/doc-update` setelah:
1. Menambahkan script Python baru ke project
2. Mengubah argumen CLI (`argparse`) di script yang sudah ada
3. Mengubah perilaku default sebuah command
4. Menambahkan atau mengubah slash command di `.claude/commands/`

Urutan kerja:
1. Tulis atau ubah kode terlebih dahulu
2. Jalankan `/doc-update` — Claude akan scan perubahan
3. Claude membandingkan kode aktual vs isi `CLAUDE.md` dan `README.md`
4. Claude mengusulkan atau langsung menulis pembaruan dokumentasi

## Implementation

Parse $ARGUMENTS untuk menentukan target (`--claude`, `--readme`, `--check`). Tanpa flag, proses keduanya.

Jika $ARGUMENTS mengandung "help" atau "--help", tampilkan usage di atas dan berhenti.

### 1. Scan file Python di root project

```
find . -maxdepth 1 -name "*.py" | sort
```

Untuk setiap file `.py`, ekstrak:
- Argumen CLI (`argparse`) — nama flag, default, help text
- Fungsi entry point utama (`if __name__ == "__main__"`)
- Docstring di bagian atas file (Usage block)
- Jika file adalah sebuah executable script CLI, maka tambahkan informasi "Alur kerja yang disarankan:"

### 2. Bandingkan dengan CLAUDE.md

Baca `CLAUDE.md` dan periksa:
- Apakah semua script `.py` di root sudah tercantum di section **Project Files**?
- Apakah contoh command di tiap section masih akurat (flag, default value)?
- Apakah ada script baru yang belum didokumentasikan?
- Apakah ada flag yang berubah atau ditambahkan?

### 3. Bandingkan dengan README.md

Baca `README.md` dan periksa:
- Apakah overview project masih akurat?
- Apakah contoh penggunaan di README masih sesuai dengan kode?
- Apakah ada fitur baru yang perlu ditambahkan ke bagian Features?

### 4. Hasilkan laporan atau tulis pembaruan

Jika `--check`:
Tampilkan laporan dalam format ini:

```
DOKUMENTASI COVERAGE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Script yang terdokumentasi di CLAUDE.md : X / Y
Script yang terdokumentasi di README.md : X / Y

PERLU DIPERBARUI
─────────────────
✗ CLAUDE.md — script_baru.py belum tercantum di Project Files
✗ CLAUDE.md — flag --tickers di ridge_config_search.py belum didokumentasikan
✗ README.md — contoh penggunaan logistic_classifier.py sudah tidak akurat

SUDAH SINKRON
──────────────
✓ CLAUDE.md — lstm_predictor.py
✓ CLAUDE.md — ridge_predictor.py
✓ README.md — overview project
```

Jika tidak ada flag `--check`, tulis langsung pembaruan ke file dokumentasi yang relevan.
