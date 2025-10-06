# Roger-Proof-RH Toolkit

Este repositório contém utilitários para executar o pipeline de validação numérica do operador Roger-Hamilton descrito nos agentes. O foco é facilitar execuções da Fase 2 (modo Schrödinger), construindo o unitário \(U = M_\Phi S_\tau\), extraindo o Hamiltoniano \(H = i\log U\) e comparando os autovalores com os zeros não triviais da função zeta de Riemann.

## Estrutura

- `roger_hamilton/`: biblioteca com componentes reutilizáveis (geração de primos, construção do unitário, análise espectral e comparação com os zeros conhecidos).
- `run_level2.py`: CLI que executa o pipeline completo, gera tabelas e gráficos (em SVG quando `matplotlib` não estiver disponível) e salva os logs exigidos pelos agentes.
- `artifacts/`: diretório padrão para saídas (pode ser alterado via `--output-dir`). Cada execução é armazenada em uma subpasta própria (`run_YYYYMMDD-HHMMSS` por padrão).

## Requisitos

- Python 3.11+
- NumPy
- (Opcional) Matplotlib – usado para gráficos quando disponível. Caso contrário, arquivos SVG são produzidos via fallback interno.

Instale as dependências mínimas com:

```bash
pip install numpy matplotlib
```

## Uso rápido

Executa a configuração de referência mencionada nos agentes:

```bash
python run_level2.py \
  --mode schrodinger \
  --N 512 --T 24 --tau 0.37 \
  --P 700 --beta 1.0 --omega 1.05 \
  --seed 42 --m 30 \
  --spacing-count 200 \
  --output-dir artifacts/reference \
  --run-name referencia
```

Saídas esperadas:

- `logs/params.json`: parâmetros utilizados na execução.
- `logs/unitarity_check.txt`: norma de \(U^\*U - I\).
- `tables/level2_m30.csv`: tabela \((\gamma_n, E_n, \Delta_n)\).
- `plots/hist_spacing.svg`, `plots/ecdf_spacing.svg`, `plots/delta_vs_n.svg`: gráficos de apoio.

> **Nota:** omitir `--run-name` cria automaticamente uma pasta com timestamp (`run_20240101-120000`, por exemplo). Use `--reset-output` para limpar completamente o diretório indicado em `--output-dir` antes de uma nova bateria de testes.

> **Nota:** o script aborta automaticamente caso \(\lVert U^\* U - I \rVert_2 > 10^{-10}\), conforme solicitado pelos agentes.

## Desenvolvimento

Para executar um teste rápido com uma malha menor:

```bash
python run_level2.py --N 128 --T 8 --P 100 --m 10 --spacing-count 50 --output-dir artifacts/smoke --run-name teste
```

Isso gera as mesmas saídas em resolução reduzida, útil para validar o ambiente antes de uma varredura completa.
