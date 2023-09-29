lf <- function(filename, mmin=-22, mmax=-15, p0=c(-2, -20.9, -1.2)) {
  zz <- file(description = filename, open = "rb")
  header0 <- readFITSheader(zz) # read primary header
  header <- readFITSheader(zz) # read extension header
  D <- readFITSbintable(zz, header)
  close(zz)

  sel <- (D$col[[2]] >= mmin) & (D$col[[2]] < mmax)
  survey = dffit(D$col[[2]][sel], D$col[[7]][sel], D$col[[3]][sel], gdf=schecMag, p.initial=p0)
#  dfwrite(survey)
  dfplot(survey)
  return(survey)
  }

schecMag <- function(M, p, output = "density") {
  if (output == "density") {
	L = 10**(0.4*(p[2] - M))
	return(0.4*log(10) * 10**p[1] * L**(1+p[3]) * exp(-L))
  }
  else if (output == "equation") {
     return("dN/(dVdx) = 0.4*log(10)*10^p[1]*mu^(p[3]+1)*exp(-mu), where mu=10^(0.4*)(p[2]-x))")
  }
  else if (output == "names") {
    names = c(expression("log"[10] * "(" * phi["*"] * ")"), 
            expression("log"[10] * "(M"["*"] * ")"), expression(alpha))
        return(names)
  }
  return(NULL)

}

survey = lf('Vmax_r.fits')
dfplot(survey, p=c(-2, -20.9, -1.2))