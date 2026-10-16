# Triagem de tuberculose em raio-x (DenseNet121)

CNN em PyTorch para classificar radiografias de tórax em:

- `parece tuberculose (sugestivo)`
- `não parece tuberculose (não sugestivo)`

O modelo final usa **máscara total do pulmão** (tudo fora do pulmão é zerado e a imagem é recortada) e uma **DenseNet121** pré-treinada no ImageNet. A região suspeita é marcada com Grad-CAM restrito ao pulmão.

## Estrutura

```text
tb_cnn/
  common.py       # leitura DICOM, máscara, modelo
  preprocess.py   # gera cache 512x512 com máscara total
  train.py        # dataset, treino e métricas
  treinar.py      # validação cruzada + modelo final
  predict.py      # analisa um exame ou uma pasta
  requirements.txt
  modelos/        # modelo_final.pt
  resultados/     # métricas e imagens do teste
DIVISAO 80-10-10/ # dados locais (não vão para o GitHub)
```

## Ambiente

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r tb_cnn\requirements.txt
```

É necessário ter a pasta `DIVISAO 80-10-10` com:

- `TREINAMENTO\COM TUBERCULOSE`
- `TREINAMENTO\SEM TUBERCULOSE`
- `TREINAMENTO\TB SCORE 25 OU MENOS (desconsiderada no treinamento)`
- `TESTE`
- `VALIDACAO ESPECIALISTAS`
- `lista_movidos.csv`

## Uso

```powershell
cd tb_cnn

# 1) Gerar cache (máscara total)
python preprocess.py

# 2) Treinar (5-fold × 3 sementes + modelo final)
python treinar.py

# 3) Analisar um exame ou pasta
python predict.py "..\DIVISAO 80-10-10\TESTE\arquivo.dcm"
python predict.py "..\DIVISAO 80-10-10\TESTE" --saida saida_predicoes
```

## Resultado do modelo final (caso máscara total)

Validação cruzada (média ± desvio de 15 treinamentos: 5 partes × 3 repetições):

| AUC | Acurácia | Sensibilidade | Especificidade |
|---|---|---|---|
| 79,2 ± 6,2 | 69,8 ± 6,1 | 69,8 ± 8,7 | 69,8 ± 16,9 |

Base de teste (49 exames, média ± desvio de 3 modelos):

| AUC | Acurácia | Sensibilidade | Especificidade |
|---|---|---|---|
| 89,9 ± 3,2 | 81,6 ± 5,4 | 86,0 ± 10,4 | 74,1 ± 17,0 |

## Observações

- As imagens DICOM originais **não** são alteradas; o preprocessamento grava apenas em `tb_cnn/cache`.
- O `.gitignore` impede o envio de pastas de pacientes (`.dcm`, `DIVISAO 80-10-10`, pastas datadas).
- O arquivo `modelos/modelo_final.pt` é a semente com maior AUC no teste.
