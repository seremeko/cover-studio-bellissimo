# AI Cover Studio (Cover Studio Bellissimo)

Applicazione desktop e web locale sviluppata in Python e Gradio per generare AI Cover vocali (Voce Singola o Duetto) interamente in locale, basata su **Demucs**, **RVC v2 (`rmvpe`)**, **PyWorld**, **Pedalboard** e **FFmpeg**.

Progetto personale di **Meko**, progettato con una pipeline audio pulita (zero AutoTune, zero effetti Chorus o vocoder applicati prima dell'inferenza RVC, allineamento rigoroso a `0.0s`) e un'interfaccia ispirata ai blog e journal kawaii dei primi anni 2000.

---

## Indice

1. [Caratteristiche principali](#caratteristiche-principali)
2. [Struttura del repository](#struttura-del-repository)
3. [Architettura della pipeline audio](#architettura-della-pipeline-audio)
4. [Interfaccia grafica e funzionalita](#interfaccia-grafica-e-funzionalita)
5. [Ottimizzazioni per esecuzione su CPU](#ottimizzazioni-per-esecuzione-su-cpu)
6. [Requisiti e installazione](#requisiti-e-installazione)
7. [Avvio rapido](#avvio-rapido)
8. [Gestione dei modelli vocali RVC v2](#gestione-dei-modelli-vocali-rvc-v2)
9. [Script standalone nella root](#script-standalone-nella-root)

---

## Caratteristiche principali

- **Esecuzione 100% locale**: nessun servizio cloud esterno o API a pagamento per separazione audio, inferenza vocale o mixaggio.
- **Download diretto da YouTube**: accetta sia un URL YouTube diretto sia un semplice titolo canzone (risolto automaticamente tramite `ytsearch1:`).
- **Separazione stem professionale**: estrazione di voce (`vocals.wav`) e base strumentale (`instrumental.wav`) mediante Demucs (`htdemucs`, `--two-stems=vocals`).
- **Duetto automatico via analisi F0**: in modalita Duetto, analizza il contorno di frequenza fondamentale (F0) della voce originale con PyWorld per separare automaticamente le parti maschili da quelle femminili, applicando crossfade da `100ms` e gating morbido da `50ms`.
- **Inferenza RVC v2 pura**: utilizza `rvc_cli` (motore Applio/RVC) con estrattore `rmvpe`, `embedder_model="contentvec"`, `protect=0.33` e `f0_autotune=False`.
- **Pitch automatico intelligente**: in modalita `Auto`, calcola la mediana F0 dei frame vocali attivi e seleziona automaticamente `0`, `+12` o `-12` semitoni in base al registro di partenza e al modello di destinazione.
- **Downloader multi-fonte integrato**: cerca su Hugging Face Hub (Models e Datasets), GitHub e URL diretti, scarica, estrae (anche da `.zip`) e verifica tramite `torch.load` che il checkpoint sia effettivamente un modello RVC v2 valido con `f0` attivo prima di aggiungerlo ai menu.
- **Training locale RVC v2 e Text to Speech**: tab dedicate per addestrare nuovi modelli vocali RVC v2 dai propri file audio e per generare tracce parlate con `edge-tts` convertite con la voce scelta.
- **Mixaggio e Mastering da studio**: catena `pedalboard` con filtro passa-alto, compressione leggera, riverbero Plate caldo attivabile/disattivabile, calibrazione automatica del livello RMS rispetto alla voce originale, limiter sul master ed esportazione finale in **MP3 a 320 kbps** (`44100 Hz` stereo) tramite `ffmpeg-python`.
- **Lettore MP3 pixel art integrato**: mini-player kawaii posizionato sopra la sidebar sinistra con riproduzione, regolazione volume (`VOL`), menu tracce con download diretto per ogni MP3 e aggiornamento automatico a ogni nuova generazione.

---

## Struttura del repository

```text
COVER-STUDIO-BELLISSIMO/
|-- AI_Cover_Studio/
|   |-- app.py                  # Applicazione principale Gradio + pipeline backend
|   |-- make_sprites.py         # Generatore degli asset pixel art locali (sfondo e sprite)
|   |-- avvia_studio.bat        # Script di avvio rapido per Windows
|   |-- assets/                 # Sfondo pixel art (bg_sky.png), sprite shimeji e rika_nipah.mp3
|   |-- datasets/               # (Ignorata da Git) Dataset audio per il training locale
|   `-- output/                 # (Ignorata da Git) Cartelle di lavoro per ogni canzone e TTS
|-- rvc_cli/                    # Motore RVC v2 (inferenza, RMVPE, ContentVec, config)
|-- models/                     # (Ignorata da Git) Modelli RVC locali (.pth + .index)
|-- phase1_download.py          # Script standalone: download audio yt-dlp
|-- phase2_demucs.py            # Script standalone: separazione stem Demucs
|-- phase3_search_hf.py         # Script standalone: ricerca modelli su Hugging Face
|-- phase3_download_model.py    # Script standalone: download ed estrazione modelli
|-- phase5_rvc_infer.py         # Script standalone: inferenza RVC v2 singola
|-- phase6_mix.py               # Script standalone: mixaggio duet + Pedalboard + MP3
|-- requirements.txt            # Dipendenze Python del progetto
`-- .gitignore                  # Esclude venv, modelli, pesi binari, audio e output
```

*(L'ambiente virtuale `venv/` viene rilevato automaticamente sia dentro `COVER-STUDIO-BELLISSIMO/venv/` sia nella cartella superiore `../venv/`.)*

---

## Architettura della pipeline audio

Ogni generazione eseguita da `AI_Cover_Studio/app.py` segue cinque fasi sequenziali nella cartella dedicata `AI_Cover_Studio/output/<nome_canzone>/`:

### Step 1 - Acquisizione audio (`yt-dlp`)
- Se l'input inizia con `http://`, `https://` o contiene `youtube.com` / `youtu.be`, viene trattato come URL diretto; altrimenti viene cercato il primo risultato su YouTube tramite `ytsearch1:<titolo>`.
- Il titolo del brano viene risolto e sanitizzato per creare la directory di lavoro `output/<nome_canzone>/`.
- L'audio viene estratto e convertito in `original.wav`.
- Se `original.wav` esiste gia ed e valido (`> 10 KB`), il download viene saltato tramite cache locale.

### Step 2 - Separazione voce e base (`Demucs`)
- Esegue `demucs --two-stems=vocals -n htdemucs -j <cpu_count>` su `original.wav`.
- Copia gli stem risultanti in `vocals.wav` e `instrumental.wav`.
- Se entrambi gli stem esistono gia nella cartella di lavoro, Demucs viene saltato automaticamente per velocizzare i tentativi successivi sulla stessa canzone.

### Step 3 - Modalita Voce Singola o Duetto (Split F0)
- **Voce Singola**: `vocals.wav` passa direttamente allo Step 4 senza alcuna alterazione preliminare (nessun Chorus o vocoder pre-RVC).
- **Duetto (Split Maschile / Femminile via F0)**:
  - Analizza `vocals.wav` in mono con `pyworld.dio` + `pyworld.stonemask` (`frame_period=20ms`, `f0_floor=65 Hz`, `f0_ceil=600 Hz`).
  - Calcola l'energia RMS per frame per escludere respiri e rumore di fondo (`f0 > 70 Hz` e `rms > 0.008`).
  - Determina la soglia femminile adattiva `max(210.0, global_med * 1.40)` su finestre mobili da `240ms` (passo `120ms`).
  - Unisce intervalli separati da pause `<= 450ms`, scarta frammenti `< 300ms` e applica un margine di `120ms`.
  - Genera due tracce complementari di lunghezza identica (`vocals_male.wav` e `vocals_female.wav`) con `pydub` e crossfade da `100ms` sulle transizioni.

### Step 4 - Inferenza RVC v2 pura (`rvc_cli`)
- Richiama `core.run_infer_script` nella directory `rvc_cli/` con i parametri:
  - `f0_method="rmvpe"`
  - `protect=0.33`
  - `index_rate` configurabile (default `0.75`)
  - `volume_envelope=1.0`
  - `f0_autotune=False`
  - `embedder_model="contentvec"`
  - `export_format="WAV"`
- Se il pitch e impostato su `Auto`, `decide_auto_pitch()` stima la mediana F0 della traccia e seleziona `0`, `+12` o `-12`.
- L'inferenza e avvolta da `run_rvc_active_only()`, che elabora solo le regioni non silenziose e ricostruisce la timeline esatta.

### Step 5 - Mixaggio e Mastering (`Pedalboard` + `ffmpeg-python`)
- Carica base strumentale, voce originale e tracce convertite a `44100 Hz` stereo, troncandole alla lunghezza minima comune e mantenendo l'allineamento rigoroso a `0.0s`.
- In modalita Duetto applica una maschera morbida (`50ms` di convoluzione) per silenziare il rumore di fondo di ciascuna traccia RVC durante i turni dell'altro cantante.
- Catena effetti per ogni traccia vocale:
  - `HighpassFilter(110 Hz)`
  - `Compressor(threshold_db=-18.0, ratio=2.8, attack_ms=12.0, release_ms=130.0)`
  - `Reverb(room_size=0.60, damping=0.56, wet_level=0.22, dry_level=0.89, width=1.0)` (se **Riverbero Plate** e attivo)
- Calibrazione del livello vocale: normalizza l'RMS attivo della voce convertita a `0.95 * orig_rms` (con `+1.2 dB` circa sulla seconda voce nel duetto), poi somma `instrumental * 0.90`.
- Master bus: `Compressor(threshold_db=-12.0, ratio=1.8)` seguito da `Limiter(threshold_db=-0.8 dB)`.
- Esporta `<nome_canzone>_master.wav` e lo converte in `<nome_canzone>_cover.mp3` (`320 kbps`, `libmp3lame`, `44100 Hz`).

---

## Interfaccia grafica e funzionalita

L'interfaccia Web in Gradio e strutturata come un blog/journal dei primi anni 2000:

- **Sfondo pixel art**: cielo pastello con nuvole e cityscape pixelato (`assets/bg_sky.png`) visibile dietro al contenitore trasparente.
- **Lettore MP3 pixel art (colonna sinistra)**: widget compatto posizionato sopra la sidebar informativa, con ciondolo a cuoricino pixelato, CD olografico rotante, barra di avanzamento a 14 tacche cliccabile, pulsanti `<<`, Play/Pausa, `>>`, controllo volume (`VOL` con slider `0-100%`) e menu a tre linee per sfogliare tutte le cover/TTS e scaricarle singolarmente. Si aggiorna automaticamente al termine di ogni nuova generazione.
- **Sidebar guida di Meko**: pannello sinistro sotto il player con le sezioni `ABOUT`, `HOW IT WORKS`, `MADE WITH` e `STATUS` (conteggio dei modelli locali pronti).
- **Card principale a righe da quaderno**: griglia da `28px` allineata alle intestazioni con quattro tab su una sola riga:
  1. **Generatore Cover**: input brano/URL, modalita Singola o Duetto, selezione modelli con tasto `Aggiorna lista`, selettore Pitch (`Auto`, `-12`, `0`, `+12`, `Manuale`), slider `Index Rate`, checkbox `Riverbero Plate`, log testuale live, barra di avanzamento e player audio finale.
  2. **Scarica Nuova Voce**: ricerca multi-fonte (Hugging Face Models/Datasets, GitHub, URL diretto) per nome personaggio/artista, download automatico in `models/<nome>`, verifica RVC v2 e aggiornamento immediato dei menu a tendina.
  3. **Crea Voce**: addestramento locale di un nuovo modello RVC v2 a partire dai propri file audio.
  4. **Text to Speech**: sintesi vocale con `edge-tts` e conversione con il modello RVC v2 selezionato.
- **Mascotte interattive pixel art**: sprite scontornati (`Miku`, `Gumi`, `Rin`, `Len`, `Luka`, `Kaito`, `Rika`, `Rena`, `Yoru`, `Asa`, `Teto`) tutti sulla stessa linea e proporzione, con fisica in stile Shimeji:
  - Camminano autonomamente lungo il bordo inferiore della finestra.
  - Possono essere afferrati e trascinati ovunque con il mouse (`pointerdown` / `pointermove`).
  - Al rilascio cadono verso il basso con gravita e rimbalzo morbido.
  - Al click eseguono un salto con rotazione; al doppio click cambiano direzione con un salto alto.
  - Negli urti ad alta velocita riproducono effetti sonori dedicati (inclusa la voce `"nipah"` per Rika sincronizzata con l'impatto).

---

## Ottimizzazioni per esecuzione su CPU

Il progetto e ottimizzato per funzionare anche su macchine prive di GPU NVIDIA/CUDA:

1. **Parallelismo PyTorch**: `torch.set_num_threads(os.cpu_count())` all'avvio.
2. **Parallelismo Demucs**: flag `-j <cpu_count>` per distribuire l'elaborazione tra i core disponibili.
3. **Cache automatica degli step intermedi**: `original.wav`, `vocals.wav` e `instrumental.wav` vengono riutilizzati se gia presenti in `output/<nome_canzone>/`.
4. **Inferenza RVC solo su segmenti attivi (`run_rvc_active_only`)**: analizza l'audio a finestre di `50ms` (`rms > 0.0015`), fonde pause `<= 1.2s` e converte solo le porzioni non silenziose quando il parlato/cantato occupa meno dell'85% della traccia, risparmiando molto tempo soprattutto sulle tracce complementari del Duetto.

---

## Requisiti e installazione

- **Sistema operativo**: Windows 10/11 (compatibile anche con Linux/macOS adattando il launcher e i percorsi FFmpeg).
- **Python**: consigliato Python `3.12` con ambiente virtuale `venv`.
- **Binari esterni**:
  - `ffmpeg` e `ffprobe` (supportato `static_ffmpeg` dentro il `venv` o installazione di sistema nel `PATH`).
  - I pesi base di `rvc_cli` (`rmvpe.pt` e `contentvec/pytorch_model.bin`) devono essere presenti in `rvc_cli/rvc/models/`.

### Setup da zero

```powershell
git clone https://github.com/seremeko/cover-studio-bellissimo.git
cd cover-studio-bellissimo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install gradio
```

---

## Avvio rapido

### Su Windows con doppio click
Esegui `avvia_studio.bat` (presente nella cartella `AI_Cover_Studio` o copiato sul Desktop): avvia `app.py` con flag `-u` usando il Python del `venv` e apre automaticamente il browser su `http://127.0.0.1:7860`.

### Da terminale

```powershell
cd AI_Cover_Studio
..\venv\Scripts\python.exe -u app.py
```

Al primo avvio sono necessari circa 15-20 secondi per il caricamento di `torch`, `librosa` e `gradio`.

---

## Gestione dei modelli vocali RVC v2

I modelli locali vanno inseriti nella cartella `models/` nella root del progetto, organizzati in sottocartelle contenenti almeno un file `.pth` (non `G_*` o `D_*`) e il relativo `.index`:

```text
models/
|-- miku/
|   `-- nome_modello/
|       |-- model.pth
|       `-- model.index
`-- altro_personaggio/
    |-- voce.pth
    `-- voce.index
```

- All'avvio `app.py` scansiona ricorsivamente `models/` ignorando `.cache` e `*_downloads`.
- Dopo aver aggiunto manualmente nuovi file in `models/`, premere **Aggiorna lista** nella tab **Generatore Cover**.
- In alternativa, usare la tab **Scarica Nuova Voce** per cercare e installare modelli RVC v2 direttamente da Hugging Face Hub.

---

## Script standalone nella root

Oltre alla web app `AI_Cover_Studio/app.py`, la root conserva gli script modulari originali della pipeline:

- `phase1_download.py`: download di tracce YouTube in `.wav` con `yt-dlp`.
- `phase2_demucs.py`: separazione `--two-stems=vocals` con `htdemucs`.
- `phase3_search_hf.py`: ricerca di repository RVC su Hugging Face Hub.
- `phase3_download_model.py`: download ed estrazione di archivi `.zip` da Hugging Face.
- `phase5_rvc_infer.py`: esecuzione diretta di `run_infer_script` da `rvc_cli`.
- `phase6_mix.py`: script di arrangiamento duetto con `pydub` + mastering con `pedalboard` e `ffmpeg-python`.
