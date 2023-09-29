library(FITSio)
library(dftools)

path = '~/Data/gama/lambda_kcorr/Vmax_z006' # CHANGE THIS TO YOUR DATA DIRECTORY

lf <- function(filename, mmin=-25, mmax=-15, p0=c(-2, -20.9, -1.2)) {

  # lf model
  lf.model = function(x,p) {
    L = 10^(0.4*(p[2] - x))
    return(0.4*log(10)*10^p[1]*L^(1+p[3])*exp(-L))
  }

  # read data
  zz <- file(description = filename, open = "rb")
  header0 <- readFITSheader(zz) # read primary header
  header <- readFITSheader(zz) # read extension header
  D <- readFITSbintable(zz, header)
  close(zz)
  
  # selection
  sel <- (D$col[[2]] >= mmin) & (D$col[[2]] < mmax)
  
  # fit
  survey = dffit(D$col[[2]][sel], D$col[[7]][sel], D$col[[3]][sel], gdf=lf.model, p.initial=p0, obs.selection = function(x) as.numeric(x>=mmin & x<=mmax))
  return(survey)

}

# fit LF
survey = lf(file.path(path,'Vmax_r.fits'))

# write coefficients
for (i in seq_along(survey$fit$p.best)) {
  cat(sprintf('p[%d] = %7.3f±%5.3f\n',i,survey$fit$p.best[i],survey$fit$p.sigma[i]))
}

# plot result
dfplot(survey, nbins = 30, ylim = c(1e-4,1e-1),
       xlab='Magnitude M', ylab=expression('Density per unit of M and Mpc'^3),
       show.posterior.data = FALSE, col.data.input = 'black')
legend(-20,1e-3,c('Raw binned data without uncertainties (1/Vmax)','MMLE fit accounting for magintude uncertainties'),
       lty=c(NA,1),lwd=c(1,2),pch=c(16,NA),col=c('black','blue'),bty = 'n')
