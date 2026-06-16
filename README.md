# Testare Tower-Plus-72B GGUF su P40 con offload 3050

## Descrizione
Progetto per eseguire il modello Tower-Plus-72B GGUF (4-bit) su Tesla P40 utilizzando llama.cpp, con offload della KV-cache sulla RTX 3050. Include script di avvio, test automatizzato con fallback e benchmark manuale.

## Architettura
- **GPU principale**: Tesla P40 (CUDA1) per il calcolo del modello
- **GPU secondaria**: RTX 3050 (CUDA0) per offload della KV-cache
- **Backend**: llama.cpp compilato con supporto CUDA
- **Modello**: Tower-Plus-72B GGUF 4-bit (Ultra-Uncensored-Heretic)

## Installazione
1. Clonare il repository llama.cpp con supporto CUDA
2. Compilare llama.cpp con `CUDA=1`
3. Posizionare il file GGUF del modello nella directory del progetto
4. Assicurarsi che i driver NVIDIA e la toolkit CUDA siano installati
5. Eseguire `chmod +x run_tower_72b.sh test_tower_72b.sh benchmark_tower_72b.sh`

## Uso
- Avvio diretto: `./run_tower_72b.sh`
- Test con fallback: `./test_tower_72b.sh`
- Benchmark manuale: `./benchmark_tower_72b.sh`

Gli script accettano variabili d'ambiente per configurare percorsi e parametri (vedere gli script per dettagli).

## Esempi
```bash
# Test rapido con fallback automatico
./test_tower_72b.sh

# Benchmark con carico sostenuto
./benchmark_tower_72b.sh
```

## Stato
✅ COMPLETATO — 2026-06-16
- Fase 1: Ricerca e download modello completata
- Fase 2: Script di configurazione llama.cpp creato
- Fase 3: Script test automatizzato con fallback creato
- Fase 4: Script benchmark manuale creato
- Fase 5: Documentazione vault completata