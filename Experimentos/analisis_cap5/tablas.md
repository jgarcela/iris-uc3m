## 1. B0 vs B1 por variable

### V25: lenguaje_sexista (prevalencia real = 0,810)

| Modelo | Nivel | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B0 | 1312 | 0,015 | 0,199 | 0,800 | 0,015 | 0,030 | 0,174 | -0,000 |
| gemma4:e4b | B0 | 1237 | 0,022 | 0,200 | 0,704 | 0,019 | 0,037 | 0,176 | -0,006 |
| gpt-4o-mini | B0 | 1313 | 0,090 | 0,236 | 0,754 | 0,084 | 0,151 | 0,228 | -0,013 |
| gpt-5.4-nano | B0 | 1313 | 0,026 | 0,207 | 0,824 | 0,026 | 0,051 | 0,185 | 0,001 |
| gemini-3.1-flash-lite | B1 | 1313 | 0,039 | 0,226 | 0,961 | 0,046 | 0,088 | 0,208 | 0,015 |
| gemma4:e4b | B1 | 1313 | 0,012 | 0,201 | 0,938 | 0,014 | 0,028 | 0,175 | 0,004 |
| gpt-4o-mini | B1 | 1313 | 0,462 | 0,457 | 0,789 | 0,450 | 0,573 | 0,414 | -0,037 |
| gpt-5.4-nano | B1 | 1313 | 0,001 | 0,191 | 1 | 0,001 | 0,002 | 0,161 | 0 |

### V26: masc_generico (prevalencia real = 0,810)

| Modelo | Nivel | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B0 | 1313 | 0,257 | 0,397 | 0,902 | 0,287 | 0,435 | 0,394 | 0,073 |
| gemma4:e4b | B0 | 1303 | 0,323 | 0,452 | 0,907 | 0,361 | 0,517 | 0,442 | 0,102 |
| gpt-4o-mini | B0 | 1313 | 0,179 | 0,317 | 0,855 | 0,189 | 0,309 | 0,317 | 0,023 |
| gpt-5.4-nano | B0 | 1309 | 0,554 | 0,580 | 0,854 | 0,582 | 0,692 | 0,515 | 0,099 |
| gemini-3.1-flash-lite | B1 | 1313 | 0,669 | 0,708 | 0,887 | 0,733 | 0,803 | 0,621 | 0,261 |
| gemma4:e4b | B1 | 1313 | 0,369 | 0,482 | 0,897 | 0,408 | 0,561 | 0,465 | 0,109 |
| gpt-4o-mini | B1 | 1313 | 0,191 | 0,332 | 0,873 | 0,206 | 0,333 | 0,332 | 0,034 |
| gpt-5.4-nano | B1 | 1313 | 0,097 | 0,262 | 0,874 | 0,104 | 0,186 | 0,256 | 0,016 |

### V30: sexismo_discurso (prevalencia real = 0,428)

| Modelo | Nivel | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B0 | 1312 | 0,016 | 0,571 | 0,476 | 0,018 | 0,034 | 0,379 | 0,004 |
| gemma4:e4b | B0 | 1296 | 0,039 | 0,569 | 0,440 | 0,040 | 0,073 | 0,396 | 0,002 |
| gpt-4o-mini | B0 | 1313 | 0,127 | 0,554 | 0,431 | 0,128 | 0,198 | 0,445 | 0,002 |
| gpt-5.4-nano | B0 | 1313 | 0,142 | 0,554 | 0,439 | 0,146 | 0,219 | 0,454 | 0,007 |
| gemini-3.1-flash-lite | B1 | 1313 | 0,020 | 0,569 | 0,423 | 0,020 | 0,037 | 0,380 | -0,000 |
| gemma4:e4b | B1 | 1313 | 0,028 | 0,565 | 0,378 | 0,025 | 0,047 | 0,383 | -0,006 |
| gpt-4o-mini | B1 | 1313 | 0,107 | 0,561 | 0,450 | 0,112 | 0,179 | 0,440 | 0,011 |
| gpt-5.4-nano | B1 | 1313 | 0,112 | 0,558 | 0,435 | 0,114 | 0,181 | 0,439 | 0,004 |

### V33: asimetria_mujer_hombre (prevalencia real = 0,062)

| Modelo | Nivel | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B0 | 1313 | 0,005 | 0,936 | 0,286 | 0,025 | 0,045 | 0,506 | 0,036 |
| gemma4:e4b | B0 | 1306 | 0,079 | 0,876 | 0,107 | 0,136 | 0,120 | 0,526 | 0,054 |
| gpt-4o-mini | B0 | 1312 | 0,144 | 0,816 | 0,074 | 0,173 | 0,104 | 0,500 | 0,019 |
| gpt-5.4-nano | B0 | 1306 | 0,377 | 0,617 | 0,075 | 0,457 | 0,129 | 0,442 | 0,025 |
| gemini-3.1-flash-lite | B1 | 1313 | 0,039 | 0,909 | 0,118 | 0,074 | 0,091 | 0,521 | 0,045 |
| gemma4:e4b | B1 | 1313 | 0,069 | 0,889 | 0,143 | 0,160 | 0,151 | 0,546 | 0,092 |
| gpt-4o-mini | B1 | 1313 | 0,144 | 0,813 | 0,063 | 0,148 | 0,089 | 0,492 | 0,003 |
| gpt-5.4-nano | B1 | 1313 | 0,007 | 0,931 | 0 | 0,000 | 0 | 0,482 | -0,012 |

### V35: denominacion_sexualizada (prevalencia real = 0,100)

| Modelo | Nivel | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B0 | 1313 | 0,005 | 0,898 | 0,286 | 0,015 | 0,029 | 0,488 | 0,019 |
| gemma4:e4b | B0 | 1305 | 0,033 | 0,896 | 0,395 | 0,134 | 0,200 | 0,572 | 0,159 |
| gpt-4o-mini | B0 | 1313 | 0,025 | 0,896 | 0,424 | 0,107 | 0,171 | 0,558 | 0,136 |
| gpt-5.4-nano | B0 | 1312 | 0,025 | 0,895 | 0,394 | 0,099 | 0,159 | 0,551 | 0,123 |
| gemini-3.1-flash-lite | B1 | 1313 | 0,009 | 0,894 | 0,167 | 0,015 | 0,028 | 0,486 | 0,011 |
| gemma4:e4b | B1 | 1313 | 0,014 | 0,890 | 0,158 | 0,023 | 0,040 | 0,491 | 0,015 |
| gpt-4o-mini | B1 | 1313 | 0,012 | 0,897 | 0,375 | 0,046 | 0,082 | 0,514 | 0,061 |
| gpt-5.4-nano | B1 | 1313 | 0,001 | 0,901 | 1 | 0,008 | 0,015 | 0,482 | 0,014 |


## 2. Ablaciones (B1 completo = bench) por variable

### V25: lenguaje_sexista (prevalencia real = 0,810)

| Modelo | Config. | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B1 | 1313 | 0,039 | 0,226 | 0,961 | 0,046 | 0,088 | 0,208 | 0,015 |
| gemini-3.1-flash-lite | abl_minimo | 1312 | 0,015 | 0,200 | 0,800 | 0,015 | 0,030 | 0,174 | -0,000 |
| gemini-3.1-flash-lite | abl_singuia | 1313 | 0,026 | 0,215 | 0,971 | 0,031 | 0,060 | 0,193 | 0,011 |
| gemini-3.1-flash-lite | abl_sinres | 1312 | 0,034 | 0,222 | 0,977 | 0,040 | 0,078 | 0,202 | 0,014 |
| gemma4:e4b | B1 | 1313 | 0,012 | 0,201 | 0,938 | 0,014 | 0,028 | 0,175 | 0,004 |
| gemma4:e4b | abl_minimo | 1313 | 0,011 | 0,198 | 0,857 | 0,011 | 0,022 | 0,171 | 0,001 |
| gemma4:e4b | abl_singuia | 1313 | 0,011 | 0,197 | 0,800 | 0,011 | 0,022 | 0,171 | -0,000 |
| gemma4:e4b | abl_sinres | 1313 | 0,008 | 0,194 | 0,727 | 0,008 | 0,015 | 0,167 | -0,002 |
| gpt-4o-mini | B1 | 1313 | 0,462 | 0,457 | 0,789 | 0,450 | 0,573 | 0,414 | -0,037 |
| gpt-4o-mini | abl_minimo | 1313 | 0,491 | 0,493 | 0,808 | 0,490 | 0,610 | 0,442 | -0,004 |
| gpt-4o-mini | abl_singuia | 1313 | 0,439 | 0,458 | 0,806 | 0,437 | 0,566 | 0,423 | -0,007 |
| gpt-4o-mini | abl_sinres | 1313 | 0,501 | 0,478 | 0,787 | 0,487 | 0,602 | 0,423 | -0,045 |
| gpt-5.4-nano | B1 | 1313 | 0,001 | 0,191 | 1 | 0,001 | 0,002 | 0,161 | 0 |
| gpt-5.4-nano | abl_minimo | 1313 | 0,002 | 0,192 | 1 | 0,002 | 0,004 | 0,162 | 0,001 |
| gpt-5.4-nano | abl_singuia | 1313 | 0 | 0,190 | 0 | 0,000 | 0 | 0,160 | 0 |
| gpt-5.4-nano | abl_sinres | 1313 | 0 | 0,190 | 0 | 0,000 | 0 | 0,160 | 0 |

### V26: masc_generico (prevalencia real = 0,810)

| Modelo | Config. | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B1 | 1313 | 0,669 | 0,708 | 0,887 | 0,733 | 0,803 | 0,621 | 0,261 |
| gemini-3.1-flash-lite | abl_minimo | 1312 | 0,556 | 0,646 | 0,910 | 0,625 | 0,741 | 0,591 | 0,238 |
| gemini-3.1-flash-lite | abl_singuia | 1312 | 0,634 | 0,694 | 0,898 | 0,702 | 0,788 | 0,618 | 0,264 |
| gemini-3.1-flash-lite | abl_sinres | 1312 | 0,604 | 0,662 | 0,891 | 0,664 | 0,761 | 0,593 | 0,225 |
| gemma4:e4b | B1 | 1313 | 0,369 | 0,482 | 0,897 | 0,408 | 0,561 | 0,465 | 0,109 |
| gemma4:e4b | abl_minimo | 1313 | 0,360 | 0,489 | 0,915 | 0,407 | 0,563 | 0,474 | 0,129 |
| gemma4:e4b | abl_singuia | 1313 | 0,371 | 0,492 | 0,908 | 0,415 | 0,570 | 0,475 | 0,124 |
| gemma4:e4b | abl_sinres | 1313 | 0,370 | 0,476 | 0,887 | 0,405 | 0,556 | 0,458 | 0,098 |
| gpt-4o-mini | B1 | 1313 | 0,191 | 0,332 | 0,873 | 0,206 | 0,333 | 0,332 | 0,034 |
| gpt-4o-mini | abl_minimo | 1313 | 0,701 | 0,644 | 0,824 | 0,713 | 0,765 | 0,518 | 0,051 |
| gpt-4o-mini | abl_singuia | 1313 | 0,256 | 0,362 | 0,836 | 0,264 | 0,401 | 0,359 | 0,020 |
| gpt-4o-mini | abl_sinres | 1313 | 0,484 | 0,506 | 0,827 | 0,494 | 0,619 | 0,460 | 0,032 |
| gpt-5.4-nano | B1 | 1313 | 0,097 | 0,262 | 0,874 | 0,104 | 0,186 | 0,256 | 0,016 |
| gpt-5.4-nano | abl_minimo | 1313 | 0,099 | 0,263 | 0,869 | 0,106 | 0,189 | 0,257 | 0,016 |
| gpt-5.4-nano | abl_singuia | 1312 | 0,075 | 0,253 | 0,919 | 0,086 | 0,157 | 0,243 | 0,022 |
| gpt-5.4-nano | abl_sinres | 1311 | 0,112 | 0,264 | 0,830 | 0,115 | 0,202 | 0,259 | 0,006 |

### V30: sexismo_discurso (prevalencia real = 0,428)

| Modelo | Config. | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B1 | 1313 | 0,020 | 0,569 | 0,423 | 0,020 | 0,037 | 0,380 | -0,000 |
| gemini-3.1-flash-lite | abl_minimo | 1311 | 0,011 | 0,572 | 0,467 | 0,013 | 0,024 | 0,375 | 0,002 |
| gemini-3.1-flash-lite | abl_singuia | 1313 | 0,017 | 0,572 | 0,500 | 0,020 | 0,038 | 0,381 | 0,006 |
| gemini-3.1-flash-lite | abl_sinres | 1312 | 0,024 | 0,567 | 0,406 | 0,023 | 0,044 | 0,382 | -0,002 |
| gemma4:e4b | B1 | 1313 | 0,028 | 0,565 | 0,378 | 0,025 | 0,047 | 0,383 | -0,006 |
| gemma4:e4b | abl_minimo | 1313 | 0,035 | 0,563 | 0,370 | 0,030 | 0,056 | 0,386 | -0,009 |
| gemma4:e4b | abl_singuia | 1313 | 0,030 | 0,570 | 0,475 | 0,034 | 0,063 | 0,392 | 0,007 |
| gemma4:e4b | abl_sinres | 1313 | 0,037 | 0,561 | 0,354 | 0,030 | 0,056 | 0,385 | -0,012 |
| gpt-4o-mini | B1 | 1313 | 0,107 | 0,561 | 0,450 | 0,112 | 0,179 | 0,440 | 0,011 |
| gpt-4o-mini | abl_minimo | 1313 | 0,135 | 0,558 | 0,446 | 0,141 | 0,214 | 0,453 | 0,011 |
| gpt-4o-mini | abl_singuia | 1313 | 0,144 | 0,550 | 0,423 | 0,142 | 0,213 | 0,449 | -0,003 |
| gpt-4o-mini | abl_sinres | 1313 | 0,130 | 0,561 | 0,456 | 0,139 | 0,213 | 0,454 | 0,016 |
| gpt-5.4-nano | B1 | 1313 | 0,112 | 0,558 | 0,435 | 0,114 | 0,181 | 0,439 | 0,004 |
| gpt-5.4-nano | abl_minimo | 1313 | 0,131 | 0,552 | 0,424 | 0,130 | 0,199 | 0,444 | -0,002 |
| gpt-5.4-nano | abl_singuia | 1311 | 0,098 | 0,566 | 0,469 | 0,107 | 0,174 | 0,440 | 0,018 |
| gpt-5.4-nano | abl_sinres | 1311 | 0,164 | 0,568 | 0,488 | 0,187 | 0,271 | 0,482 | 0,044 |

### V33: asimetria_mujer_hombre (prevalencia real = 0,062)

| Modelo | Config. | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B1 | 1313 | 0,039 | 0,909 | 0,118 | 0,074 | 0,091 | 0,521 | 0,045 |
| gemini-3.1-flash-lite | abl_minimo | 1311 | 0,037 | 0,910 | 0,122 | 0,074 | 0,092 | 0,522 | 0,048 |
| gemini-3.1-flash-lite | abl_singuia | 1312 | 0,049 | 0,900 | 0,109 | 0,086 | 0,097 | 0,522 | 0,044 |
| gemini-3.1-flash-lite | abl_sinres | 1312 | 0,040 | 0,909 | 0,135 | 0,086 | 0,105 | 0,529 | 0,060 |
| gemma4:e4b | B1 | 1313 | 0,069 | 0,889 | 0,143 | 0,160 | 0,151 | 0,546 | 0,092 |
| gemma4:e4b | abl_minimo | 1313 | 0,065 | 0,896 | 0,174 | 0,185 | 0,180 | 0,562 | 0,124 |
| gemma4:e4b | abl_singuia | 1313 | 0,059 | 0,898 | 0,156 | 0,148 | 0,152 | 0,549 | 0,098 |
| gemma4:e4b | abl_sinres | 1313 | 0,070 | 0,883 | 0,109 | 0,123 | 0,116 | 0,527 | 0,054 |
| gpt-4o-mini | B1 | 1313 | 0,144 | 0,813 | 0,063 | 0,148 | 0,089 | 0,492 | 0,003 |
| gpt-4o-mini | abl_minimo | 1313 | 0,162 | 0,796 | 0,061 | 0,160 | 0,088 | 0,487 | -0,001 |
| gpt-4o-mini | abl_singuia | 1313 | 0,188 | 0,781 | 0,081 | 0,247 | 0,122 | 0,498 | 0,032 |
| gpt-4o-mini | abl_sinres | 1312 | 0,248 | 0,726 | 0,074 | 0,296 | 0,118 | 0,478 | 0,021 |
| gpt-5.4-nano | B1 | 1313 | 0,007 | 0,931 | 0 | 0,000 | 0 | 0,482 | -0,012 |
| gpt-5.4-nano | abl_minimo | 1313 | 0,007 | 0,931 | 0 | 0,000 | 0 | 0,482 | -0,012 |
| gpt-5.4-nano | abl_singuia | 1311 | 0,007 | 0,933 | 0,111 | 0,012 | 0,022 | 0,494 | 0,010 |
| gpt-5.4-nano | abl_sinres | 1311 | 0,002 | 0,936 | 0 | 0,000 | 0 | 0,483 | -0,004 |

### V35: denominacion_sexualizada (prevalencia real = 0,100)

| Modelo | Config. | N | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | B1 | 1313 | 0,009 | 0,894 | 0,167 | 0,015 | 0,028 | 0,486 | 0,011 |
| gemini-3.1-flash-lite | abl_minimo | 1311 | 0,006 | 0,898 | 0,250 | 0,015 | 0,029 | 0,488 | 0,018 |
| gemini-3.1-flash-lite | abl_singuia | 1312 | 0,005 | 0,902 | 0,571 | 0,031 | 0,058 | 0,503 | 0,049 |
| gemini-3.1-flash-lite | abl_sinres | 1312 | 0,007 | 0,896 | 0,222 | 0,015 | 0,029 | 0,487 | 0,016 |
| gemma4:e4b | B1 | 1313 | 0,014 | 0,890 | 0,158 | 0,023 | 0,040 | 0,491 | 0,015 |
| gemma4:e4b | abl_minimo | 1313 | 0,016 | 0,896 | 0,381 | 0,061 | 0,105 | 0,525 | 0,080 |
| gemma4:e4b | abl_singuia | 1313 | 0,023 | 0,891 | 0,300 | 0,069 | 0,112 | 0,527 | 0,078 |
| gemma4:e4b | abl_sinres | 1313 | 0,017 | 0,893 | 0,273 | 0,046 | 0,078 | 0,511 | 0,051 |
| gpt-4o-mini | B1 | 1313 | 0,012 | 0,897 | 0,375 | 0,046 | 0,082 | 0,514 | 0,061 |
| gpt-4o-mini | abl_minimo | 1313 | 0,014 | 0,896 | 0,368 | 0,053 | 0,093 | 0,519 | 0,070 |
| gpt-4o-mini | abl_singuia | 1312 | 0,018 | 0,894 | 0,333 | 0,061 | 0,103 | 0,523 | 0,075 |
| gpt-4o-mini | abl_sinres | 1312 | 0,039 | 0,884 | 0,294 | 0,115 | 0,165 | 0,551 | 0,115 |
| gpt-5.4-nano | B1 | 1313 | 0,001 | 0,901 | 1 | 0,008 | 0,015 | 0,482 | 0,014 |
| gpt-5.4-nano | abl_minimo | 1313 | 0,002 | 0,899 | 0 | 0,000 | 0 | 0,473 | -0,003 |
| gpt-5.4-nano | abl_singuia | 1311 | 0,002 | 0,901 | 0,667 | 0,015 | 0,030 | 0,489 | 0,025 |
| gpt-5.4-nano | abl_sinres | 1311 | 0,007 | 0,896 | 0,222 | 0,015 | 0,029 | 0,487 | 0,016 |


## 3. Kappa entre modelos por variable

### Nivel B0

**V25: lenguaje_sexista**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1312 | 0,904 | 0,063 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1312 | 0,962 | 0,056 |
| gemini-3.1-flash-lite | gemma4:e4b | 1236 | 0,965 | 0,027 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,909 | 0,177 |
| gpt-4o-mini | gemma4:e4b | 1237 | 0,903 | 0,072 |
| gpt-5.4-nano | gemma4:e4b | 1237 | 0,957 | 0,080 |

**V26: masc_generico**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,704 | 0,139 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1309 | 0,549 | 0,142 |
| gemini-3.1-flash-lite | gemma4:e4b | 1303 | 0,695 | 0,262 |
| gpt-4o-mini | gpt-5.4-nano | 1309 | 0,553 | 0,164 |
| gpt-4o-mini | gemma4:e4b | 1303 | 0,661 | 0,122 |
| gpt-5.4-nano | gemma4:e4b | 1299 | 0,546 | 0,123 |

**V30: sexismo_discurso**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1312 | 0,874 | 0,092 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1312 | 0,859 | 0,080 |
| gemini-3.1-flash-lite | gemma4:e4b | 1295 | 0,962 | 0,274 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,895 | 0,550 |
| gpt-4o-mini | gemma4:e4b | 1296 | 0,867 | 0,104 |
| gpt-5.4-nano | gemma4:e4b | 1296 | 0,857 | 0,133 |

**V33: asimetria_mujer_hombre**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1312 | 0,854 | 0,010 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1306 | 0,626 | 0,014 |
| gemini-3.1-flash-lite | gemma4:e4b | 1306 | 0,922 | 0,063 |
| gpt-4o-mini | gpt-5.4-nano | 1305 | 0,666 | 0,190 |
| gpt-4o-mini | gemma4:e4b | 1305 | 0,831 | 0,154 |
| gpt-5.4-nano | gemma4:e4b | 1299 | 0,651 | 0,124 |

**V35: denominacion_sexualizada**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,973 | 0,092 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1312 | 0,971 | 0,042 |
| gemini-3.1-flash-lite | gemma4:e4b | 1305 | 0,963 | 0,031 |
| gpt-4o-mini | gpt-5.4-nano | 1312 | 0,966 | 0,316 |
| gpt-4o-mini | gemma4:e4b | 1305 | 0,959 | 0,250 |
| gpt-5.4-nano | gemma4:e4b | 1304 | 0,955 | 0,169 |

### Nivel B1

**V25: lenguaje_sexista**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,558 | 0,047 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1313 | 0,960 | -0,001 |
| gemini-3.1-flash-lite | gemma4:e4b | 1313 | 0,952 | 0,042 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,539 | 0,002 |
| gpt-4o-mini | gemma4:e4b | 1313 | 0,543 | 0,012 |
| gpt-5.4-nano | gemma4:e4b | 1313 | 0,987 | -0,001 |

**V26: masc_generico**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,429 | 0,055 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1313 | 0,371 | 0,012 |
| gemini-3.1-flash-lite | gemma4:e4b | 1313 | 0,559 | 0,190 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,767 | 0,071 |
| gpt-4o-mini | gemma4:e4b | 1313 | 0,640 | 0,140 |
| gpt-5.4-nano | gemma4:e4b | 1313 | 0,625 | 0,047 |

**V30: sexismo_discurso**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,880 | 0,015 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1313 | 0,877 | 0,037 |
| gemini-3.1-flash-lite | gemma4:e4b | 1313 | 0,964 | 0,236 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,864 | 0,300 |
| gpt-4o-mini | gemma4:e4b | 1313 | 0,885 | 0,107 |
| gpt-5.4-nano | gemma4:e4b | 1313 | 0,874 | 0,055 |

**V33: asimetria_mujer_hombre**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,843 | 0,086 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1313 | 0,956 | 0,022 |
| gemini-3.1-flash-lite | gemma4:e4b | 1313 | 0,924 | 0,259 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,858 | 0,048 |
| gpt-4o-mini | gemma4:e4b | 1313 | 0,826 | 0,102 |
| gpt-5.4-nano | gemma4:e4b | 1313 | 0,927 | 0,028 |

**V35: denominacion_sexualizada**

| Modelo A | Modelo B | N | Acuerdo bruto | Kappa |
|---|---|---|---|---|
| gemini-3.1-flash-lite | gpt-4o-mini | 1313 | 0,980 | 0,062 |
| gemini-3.1-flash-lite | gpt-5.4-nano | 1313 | 0,990 | -0,001 |
| gemini-3.1-flash-lite | gemma4:e4b | 1313 | 0,978 | 0,054 |
| gpt-4o-mini | gpt-5.4-nano | 1313 | 0,987 | -0,001 |
| gpt-4o-mini | gemma4:e4b | 1313 | 0,978 | 0,160 |
| gpt-5.4-nano | gemma4:e4b | 1313 | 0,985 | -0,001 |


## 4. Voto por mayoría (4 modelos, B1)

### V25: lenguaje_sexista (prevalencia real = 0,810)

| Sistema | Regla empate | Empates | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| MEJOR INDIVIDUAL: gpt-4o-mini | - | -- | 0,462 | 0,457 | 0,789 | 0,450 | 0,573 | 0,414 | -0,037 |
| voto_mayoria_4 | empate=Si | 46 | 0,037 | 0,222 | 0,938 | 0,042 | 0,081 | 0,204 | 0,012 |
| voto_mayoria_4 | empate=No | 46 | 0,002 | 0,192 | 1 | 0,002 | 0,004 | 0,162 | 0,001 |

### V26: masc_generico (prevalencia real = 0,810)

| Sistema | Regla empate | Empates | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| MEJOR INDIVIDUAL: gemini-3.1-flash-lite | - | -- | 0,669 | 0,708 | 0,887 | 0,733 | 0,803 | 0,621 | 0,261 |
| voto_mayoria_4 | empate=Si | 370 | 0,401 | 0,522 | 0,914 | 0,452 | 0,605 | 0,499 | 0,148 |
| voto_mayoria_4 | empate=No | 370 | 0,119 | 0,296 | 0,949 | 0,139 | 0,243 | 0,293 | 0,045 |

### V30: sexismo_discurso (prevalencia real = 0,428)

| Sistema | Regla empate | Empates | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| MEJOR INDIVIDUAL: gpt-5.4-nano | - | -- | 0,112 | 0,558 | 0,435 | 0,114 | 0,181 | 0,439 | 0,004 |
| voto_mayoria_4 | empate=Si | 61 | 0,055 | 0,566 | 0,444 | 0,057 | 0,101 | 0,407 | 0,004 |
| voto_mayoria_4 | empate=No | 61 | 0,008 | 0,571 | 0,455 | 0,009 | 0,017 | 0,372 | 0,001 |

### V33: asimetria_mujer_hombre (prevalencia real = 0,062)

| Sistema | Regla empate | Empates | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| MEJOR INDIVIDUAL: gemma4:e4b | - | -- | 0,069 | 0,889 | 0,143 | 0,160 | 0,151 | 0,546 | 0,092 |
| voto_mayoria_4 | empate=Si | 43 | 0,040 | 0,911 | 0,154 | 0,099 | 0,120 | 0,537 | 0,076 |
| voto_mayoria_4 | empate=No | 43 | 0,007 | 0,935 | 0,222 | 0,025 | 0,044 | 0,505 | 0,033 |

### V35: denominacion_sexualizada (prevalencia real = 0,100)

| Sistema | Regla empate | Empates | Prev. pred. | Exactitud | Precisión | Recall | F1 (Sí) | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| MEJOR INDIVIDUAL: gpt-4o-mini | - | -- | 0,012 | 0,897 | 0,375 | 0,046 | 0,082 | 0,514 | 0,061 |
| voto_mayoria_4 | empate=Si | 2 | 0,002 | 0,899 | 0,333 | 0,008 | 0,015 | 0,481 | 0,011 |
| voto_mayoria_4 | empate=No | 2 | 0,001 | 0,899 | 0 | 0,000 | 0 | 0,474 | -0,002 |

