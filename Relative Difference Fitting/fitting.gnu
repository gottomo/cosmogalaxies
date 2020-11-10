set xlabel "redshift z"
set ylabel "relative number of galaxies"

plot 'RelDif_constPhiAlpha_zref3_m28_LCDM.txt' us 1:2 title 'data'
f(x)=a*(1+x)**m*exp(b*(1+x))

# Fit for all range
fit f(x) "RelDif_constPhiAlpha_zref3_m28_LCDM.txt" us 1:2 via a, m, b
plot f(x) title "a (1+z)^m e^(b (1+z))", "RelDif_constPhiAlpha_zref3_m28_LCDM.txt" us 1:2 title "data"

# Fit for 0.5<z
fit[0.5:] f(x) "RelDif_constPhiAlpha_zref3_m28_LCDM.txt" us 1:2 via a, m, b
plot[0.5:] f(x) title "a (1+z)^m e^(b (1+z))", "RelDif_constPhiAlpha_zref3_m28_LCDM.txt" us 1:2 title "data"

# Fit for 0.8<z
fit[0.8:] f(x) "RelDif_constPhiAlpha_zref3_m28_LCDM.txt" us 1:2 via a, m, b
plot[0.8:] f(x) title "a (1+z)^m e^(b (1+z))", "RelDif_constPhiAlpha_zref3_m28_LCDM.txt" us 1:2 title "data"