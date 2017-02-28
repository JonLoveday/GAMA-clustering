/* Spatial 2-point correlation function

Revision history

1.0 05-dec-07  Original C version, based on xi.f
1.1 29-jul-08  Data pre-gridded into coarse cells of side rmax to speed up calc
               Make jackknife estimates along with full sample.
1.2 30-jul-10  Allow log or lin bins for xi(r_p, pi).
               Use Landy-Szalay estimator.
	       Fix abs bug (need fabs for non-integer).
1.3 23-apr-12  Output xi(s) and xi(r_p, pi) for all jackknife regions
               to allow error estimates on w_p(r_p) and xi(r) 
	       and full covariance analysis.
1.4 17-may-12  Add vmax parameter for each object and weight pair counts by
               1.0/min(Vmax[i], Vmax[j]).
               Binning now specified independently in r_p and pi directions.
	       Outputs pair counts as well as xi and xi jacknife estimates.
1.5 14-may-14  Fix bug in xi(r_p, pi) binning.
1.6 19-may-14  Calculate pair-weighted mean separations within each bin.
2.0 01-aug-14  Now reads density from input files and calculates min-variance 
               weight according to pair separation, i.e. J3 -> J3(s).
2.1 22-jun-15  Calculate los and perp separations using more accurate 
               Fisher+ 1994 formulae.
	       Count pairs in both log and linear rp bins.
2.2 15-sep-15  For efficiency and flexibility, calculate only one of DD, DR, RR.
2.3 17-jun-16  Add log-log binned counts; rename variables and make more global.

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

typedef struct obj {
  float x, y, z, weight, den, Vmax;
  int ireg;
} OBJ;

typedef struct cell {
  int ix, iy, iz, nobj;
  OBJ *obj;
} CELL;

int autoCorr(int ncell, CELL *cell);

int crossCorr(int ncell, CELL *galcell, int ncellr, CELL *rancell);

int countSep(OBJ obj1, OBJ obj2);

double minVarWt(double s, double den);

/* Global variables */
int nlog, nlin, njack;
float logmin, logmax, logstep, linmin, linmax, linstep, xmax, theta_max,
  J3_gamma, J3_r0, J3_rmax;
double *pc_log, *s_log, *pc_lin_lin, *pc_log_lin, *pc_log_log, 
  *rp_lin_lin, *rp_log_lin, *rp_log_log, *pi_lin_lin, *pi_log_lin, *pi_log_log;

int main(int argc, char *argv[])
{
  const char *ver = "xi 2.3";
  char *cmd = "Usage: xi <infile> <outfile> OR xi <infile> <ranfile> <outfile>";

  CELL *galcell, *rancell;
  int ngal, nran=0, nobj, nc, ncr, ncell, ncellr, icell, ix, iy, iz, 
    ijack, njackr, i, j, k, ibin, ireg;
  float x, y, z, weight, den, Vmax, cellsize, cellsizer;
  const int nameLength = 80, infoLength=512;
  char inFile[80], ranFile[80], outFile[80], 
    info_gal[infoLength], info_ran[infoLength], line[infoLength];
  FILE *file;

  if (argc < 3 || argc > 4) {
    printf("%s\n", cmd);
    return 1;
  } 
  strncpy(inFile, argv[1], nameLength);
  if (argc == 3) {
    strncpy(outFile, argv[2], nameLength);
  } else {
    strncpy(ranFile, argv[2], nameLength);
    strncpy(outFile, argv[3], nameLength);
  }

  printf("%s\n", ver);

  /* Read catalogue data (x,y,z,w,den,Vmax,ireg) */
  if ((file = fopen(inFile, "r")) == NULL) {
    printf("Error opening %s\n", inFile);
    return 1;
  }
  fgets(info_gal, infoLength, file);
  fgets(line, infoLength, file);
  sscanf(line, "%d %d %d %d %f %f %f %d %f %f %d %f %f %f %f", 
	 &ngal, &nc, &ncell, &njack, &cellsize, &logmin, &logmax, &nlog,
	 &linmin, &linmax, &nlin, &theta_max, &J3_gamma, &J3_r0, &J3_rmax);
  theta_max *= M_PI/180.0;
  galcell = malloc(ncell*sizeof(CELL));
  ngal = 0;
  for (icell = 0; icell < ncell; icell++) {
    fgets(line, infoLength, file);
    sscanf(line, "%d %d %d %d", &ix, &iy, &iz, &nobj);
    printf("nobj %d\n", nobj);
    ngal += nobj;
    galcell[icell].ix = ix;
    galcell[icell].iy = iy;
    galcell[icell].iz = iz;
    galcell[icell].nobj = nobj;
    galcell[icell].obj = malloc(nobj*sizeof(OBJ));
    for (i = 0; i < nobj; i++) {
      fgets(line, infoLength, file);
      sscanf(line, "%f %f %f %f %f %f %d", &x, &y, &z, &weight, &den, &Vmax, &ireg);
      galcell[icell].obj[i].x = x; 
      galcell[icell].obj[i].y = y; 
      galcell[icell].obj[i].z = z; 
      galcell[icell].obj[i].weight = weight; 
      galcell[icell].obj[i].den = den; 
      galcell[icell].obj[i].Vmax = Vmax; 
      galcell[icell].obj[i].ireg = ireg;
    }
  }
  fclose(file);
  printf("%d galaxies read in %d cells, njack = %d\n", ngal, ncell, njack);

  /* Bin step sizes and maximum separation for pair counting.  
     xmax is largest of logmax and linmax in linear units. */
  logstep = (logmax-logmin)/nlog;
  linstep = (linmax-linmin)/nlin;
  xmax = pow(10, logmax);
  if (linmax > xmax) xmax = linmax;
  if (xmax > cellsize) printf("Warning: max separation requested %f is larger than cellsize %f\n", xmax, cellsize);

  if (argc == 4) {
    /* Read random data (x,y,z,w,den,Vmax,ireg) */
    if ((file = fopen(ranFile, "r")) == NULL) {
      printf("Error opening %s\n", ranFile);
      return 1;
    }
    fgets(info_ran, infoLength, file);
    fgets(line, infoLength, file);
    sscanf(line, "%d %d %d %d %f", &nran, &ncr, &ncellr, &njackr, &cellsizer);

    if (cellsize != cellsizer || njack != njackr) {
      printf("Galaxy and random cellsizes %f %f or jacknife regions %d %d differ!\n", cellsize, cellsizer, njack, njackr);
      return 1;
    }

    rancell = malloc(ncellr*sizeof(CELL));
/*   for (icell = 0; icell < ncell; icell++) { */
    nran = 0;
    for (icell = 0; icell < ncellr; icell++) {
      fgets(line, infoLength, file);
      sscanf(line, "%d %d %d %d", &ix, &iy, &iz, &nobj);
      nran += nobj;
      rancell[icell].ix = ix;
      rancell[icell].iy = iy;
      rancell[icell].iz = iz;
      rancell[icell].nobj = nobj;
      rancell[icell].obj = malloc(nobj*sizeof(OBJ));
      for (i = 0; i < nobj; i++) {
	fgets(line, infoLength, file);
	sscanf(line, "%f %f %f %f %f %f %d", &x, &y, &z, &weight, &den, &Vmax, &ireg);
	rancell[icell].obj[i].x = x; 
	rancell[icell].obj[i].y = y; 
	rancell[icell].obj[i].z = z; 
	rancell[icell].obj[i].weight = weight; 
	rancell[icell].obj[i].den = den; 
	rancell[icell].obj[i].Vmax = Vmax; 
	rancell[icell].obj[i].ireg = ireg;
      }
    }
    fclose(file);
    printf("%d randoms read in %d cells, njack = %d\n", nran, ncellr, njack);
  }

  /* Allocate memory for pair counts */
  pc_log = (double*) malloc(nlog*(njack+1)*sizeof(double));
  s_log = (double*) malloc(nlog*sizeof(double));
  for (i = 0; i < nlog*(njack+1); i++) pc_log[i] = 0;
  for (i = 0; i < nlog; i++) s_log[i] = 0;

  pc_lin_lin = (double*) malloc(nlin*nlin*(njack+1)*sizeof(double));
  rp_lin_lin = (double*) malloc(nlin*nlin*sizeof(double));
  pi_lin_lin = (double*) malloc(nlin*nlin*sizeof(double));
  for (i = 0; i < nlin*nlin*(njack+1); i++) pc_lin_lin[i] = 0;
  for (i = 0; i < nlin*nlin; i++) {
    rp_lin_lin[i] = 0;
    pi_lin_lin[i] = 0;
  }

  pc_log_lin = (double*) malloc(nlog*nlin*(njack+1)*sizeof(double));
  rp_log_lin = (double*) malloc(nlog*nlin*sizeof(double));
  pi_log_lin = (double*) malloc(nlog*nlin*sizeof(double));
  for (i = 0; i < nlog*nlin*(njack+1); i++) pc_log_lin[i] = 0;
  for (i = 0; i < nlog*nlin; i++) {
    rp_log_lin[i] = 0;
    pi_log_lin[i] = 0;
  }

  pc_log_log = (double*) malloc(nlog*nlog*(njack+1)*sizeof(double));
  rp_log_log = (double*) malloc(nlog*nlog*sizeof(double));
  pi_log_log = (double*) malloc(nlog*nlog*sizeof(double));
  for (i = 0; i < nlog*nlog*(njack+1); i++) pc_log_log[i] = 0;
  for (i = 0; i < nlog*nlog; i++) {
    rp_log_log[i] = 0;
    pi_log_log[i] = 0;
  }

  printf("counting pairs ...\n");
  if (argc == 4) {
    crossCorr(ncell, galcell, ncellr, rancell);
  } else {
    autoCorr(ncell, galcell);
  }

  for (i = 0; i < nlog; i++) {
    if (pc_log[i*(njack+1)] > 0) {
      s_log[i] /= pc_log[i*(njack+1)];
    }
  }
  for (i = 0; i < nlin*nlin; i++) {
    if (pc_lin_lin[i*(njack+1)] > 0) {
      rp_lin_lin[i] /= pc_lin_lin[i*(njack+1)];
      pi_lin_lin[i] /= pc_lin_lin[i*(njack+1)];
    }
  }
  for (i = 0; i < nlog*nlin; i++) {
    if (pc_log_lin[i*(njack+1)] > 0) {
      rp_log_lin[i] /= pc_log_lin[i*(njack+1)];
      pi_log_lin[i] /= pc_log_lin[i*(njack+1)];
    }
  }
  for (i = 0; i < nlog*nlog; i++) {
    if (pc_log_log[i*(njack+1)] > 0) {
      rp_log_log[i] /= pc_log_log[i*(njack+1)];
      pi_log_log[i] /= pc_log_log[i*(njack+1)];
    }
  }

  /* Output counts to ascii file.  Header, 1d (direction averaged), 
     2d lin-lin, 2d log-lin, 2d log-log.  
     Parallel sep pi is output first and varies fastest.
  */
  if ((file = fopen(outFile, "w")) == NULL) {
    printf("Error opening %s\n", outFile);
    return 1;
  }
  fprintf(file, "%s %s\n", ver, inFile);
  fputs(info_gal, file);
  fprintf(file, "%d %d %d %d\n", ngal, nran, njack, 3);

  /* xi(s): columns are s, pc, pc[1], pc[2], ... pc[njack] */
  fprintf(file, "%d %f %f\n", nlog, logmin, logmax);
  for (i = 0; i < nlog; i++) {
    fprintf(file, "%f ", s_log[i]);
    for (ijack = 0; ijack <= njack; ijack++) {
      k = i*(njack+1) + ijack;
      fprintf(file, "%f ", pc_log[k]);
    }
    fprintf(file, "\n");
  }
  
  /* xi(r_p, pi) linear-linear binned: 
     columns are pi, r_p, pc, pc[1], pc[2], ... pc[njack] */
  fprintf(file, "%d %f %f %d %f %f\n", 
	  nlin, linmin, linmax, nlin, linmin, linmax);
  for (j = 0; j < nlin; j++) {
    for (i = 0; i < nlin; i++) {
      ibin = nlin*j + i;
      fprintf(file, "%f %f ", pi_lin_lin[ibin], rp_lin_lin[ibin]);
      for (ijack = 0; ijack <= njack; ijack++) {
	k = ibin*(njack+1) + ijack;
	fprintf(file, "%f ", pc_lin_lin[k]);
      }
      fprintf(file, "\n");
    }
  }

  /* xi(r_p, pi) log-linear binned: 
     columns are pi, r_p, pc, pc[1], pc[2], ... pc[njack] */
  fprintf(file, "%d %f %f %d %f %f\n", 
	  nlog, logmin, logmax, nlin, linmin, linmax);
  for (j = 0; j < nlog; j++) {
    for (i = 0; i < nlin; i++) {
      ibin = nlin*j + i;
      fprintf(file, "%f %f ", pi_log_lin[ibin], rp_log_lin[ibin]);
      for (ijack = 0; ijack <= njack; ijack++) {
	k = ibin*(njack+1) + ijack;
	fprintf(file, "%f ", pc_log_lin[k]);
      }
      fprintf(file, "\n");
    }
  }

  /* xi(r_p, pi) log-log binned: 
     columns are pi, r_p, pc, pc[1], pc[2], ... pc[njack] */
  fprintf(file, "%d %f %f %d %f %f\n", 
	  nlog, logmin, logmax, nlog, logmin, logmax);
  for (j = 0; j < nlog; j++) {
    for (i = 0; i < nlog; i++) {
      ibin = nlog*j + i;
      fprintf(file, "%f %f ", pi_log_log[ibin], rp_log_log[ibin]);
      for (ijack = 0; ijack <= njack; ijack++) {
	k = ibin*(njack+1) + ijack;
	fprintf(file, "%f ", pc_log_log[k]);
      }
      fprintf(file, "\n");
    }
  }

  fclose(file);

  for (icell = 0; icell < ncell; icell++) {
    free(galcell[icell].obj);
  }
  free(galcell);
  if (argc == 4) {
    for (icell = 0; icell < ncellr; icell++) {
      free(rancell[icell].obj);
    }
    free(rancell);
  }
  free(pc_log); free(pc_log_lin); free(pc_lin_lin); 
  free(s_log); free(rp_lin_lin); free(rp_log_lin); free(rp_log_log); 
  free(pi_lin_lin); free(pi_log_lin); free(pi_log_log); 

  return 0;
}

/* Auto-correlation (count forwards only).  */
int autoCorr(int ncell, CELL *cell)
{
  int i, j, icell, jcell;
  double sep;

  for (icell = 0; icell < ncell; icell++) {

    /* Objects in same cell */
    for (i = 0; i < cell[icell].nobj; i++) {
      j = i + 1;
      while (j < cell[icell].nobj) {
	sep = countSep(cell[icell].obj[i], cell[icell].obj[j]);
	j++;
      }
    }

    /* Objects in neighbouring cells */
    for (jcell = icell + 1; jcell < ncell; jcell++) {
      if (abs(cell[icell].ix - cell[jcell].ix) < 2 &&
	  abs(cell[icell].iy - cell[jcell].iy) < 2 &&
	  abs(cell[icell].iz - cell[jcell].iz) < 2) {
	for (i = 0; i < cell[icell].nobj; i++) {
	  for (j = 0; j < cell[jcell].nobj; j++) {
	    sep = countSep(cell[icell].obj[i], cell[jcell].obj[j]);
	  }
	}
      }
    }
  }
  return 0;
}

/* Cross-correlation (count both directions). */
int crossCorr(int ncell, CELL *galcell, int ncellr, CELL *rancell)
{
  int i, j, icell, jcell;
  double sep;

  for (icell = 0; icell < ncell; icell++) {
    for (jcell = 0; jcell < ncellr; jcell++) {
      if (abs(galcell[icell].ix - rancell[jcell].ix) < 2 &&
	  abs(galcell[icell].iy - rancell[jcell].iy) < 2 &&
	  abs(galcell[icell].iz - rancell[jcell].iz) < 2) {
	for (i = 0; i < galcell[icell].nobj; i++) {
	  for (j = 0; j < rancell[jcell].nobj; j++) {
	    sep = countSep(galcell[icell].obj[i], rancell[jcell].obj[j]);
	  }
	}
      }
    }
  }
  return 0;
}

/* Calculate separation between two points and increment weighted pair
   counts wp1 (direction-averaged) and wp2 (parallel and perpendicular). */

int countSep(OBJ obj1, OBJ obj2)
{
  int ibin, irp, ipi, ijack;
  double dx, dy, dz, lx, ly, lz, lnorm, dsq, sep,
    V, pairwt, s, rp, pi;

  dx = obj1.x - obj2.x;
  dy = obj1.y - obj2.y;
  dz = obj1.z - obj2.z;
  dsq = dx*dx + dy*dy + dz*dz;
  if (dsq <= xmax*xmax && dsq > 0) {

    /* xi(s) */
    s = sqrt(dsq);
    sep = log10(s);
    V = obj1.Vmax < obj2.Vmax ? obj1.Vmax : obj2.Vmax;
    pairwt = obj1.weight * minVarWt(s, obj1.den) * 
      obj2.weight* minVarWt(s, obj2.den) / V;
    if (sep >= logmin && sep < logmax) {
      ibin = (int) floor((sep - logmin)/logstep);
      if (ibin >= 0 && ibin < nlog) {
	for (ijack = 0; ijack < njack; ijack++) {
	  if ((obj1.ireg != ijack) && (obj2.ireg != ijack))
	    pc_log[ibin*(njack+1) + ijack + 1] += pairwt;
	}
	pc_log[ibin*(njack+1)] += pairwt;
	s_log[ibin] += s*pairwt;
      }
    }

    /* xi(sigma,pi)  */
    lx = 0.5*(obj1.x + obj2.x);
    ly = 0.5*(obj1.y + obj2.y);
    lz = 0.5*(obj1.z + obj2.z);
    lnorm = sqrt(lx*lx + ly*ly + lz*lz);
    pi = fabs(dx*lx + dy*ly + dz*lz)/lnorm;
    rp = sqrt(dsq - pi*pi);
    ipi = (int) floor((pi-linmin)/linstep);
    if (ipi >= 0 && ipi < nlin && rp/lnorm < theta_max) {

      /* Linear-linear */
      irp = (int) floor((rp-linmin)/linstep);
      if (irp >= 0 && irp < nlin) {
	ibin = irp*nlin + ipi;
	for (ijack = 0; ijack < njack; ijack++) {
	  if ((obj1.ireg != ijack) && (obj2.ireg != ijack))
	    pc_lin_lin[ibin*(njack+1) + ijack + 1] += pairwt;
	}
	pc_lin_lin[ibin*(njack+1)] += pairwt;
	rp_lin_lin[ibin] += rp*pairwt;
	pi_lin_lin[ibin] += pi*pairwt;
      }

      /* Log-linear */
      irp = (int) floor((log10(rp)-logmin)/logstep);
      if (irp >= 0 && irp < nlog) {
	ibin = irp*nlin + ipi;
	for (ijack = 0; ijack < njack; ijack++) {
	  if ((obj1.ireg != ijack) && (obj2.ireg != ijack))
	    pc_log_lin[ibin*(njack+1) + ijack + 1] += pairwt;
	}
	pc_log_lin[ibin*(njack+1)] += pairwt;
	rp_log_lin[ibin] += rp*pairwt;
	pi_log_lin[ibin] += pi*pairwt;
      }
    }

    /* Log-log */
    ipi = (int) floor((log10(pi)-logmin)/logstep);
    if (ipi >= 0 && ipi < nlog && rp/lnorm < theta_max) {
      irp = (int) floor((log10(rp)-logmin)/logstep);
      if (irp >= 0 && irp < nlog) {
	ibin = irp*nlog + ipi;
	for (ijack = 0; ijack < njack; ijack++) {
	  if ((obj1.ireg != ijack) && (obj2.ireg != ijack))
	    pc_log_log[ibin*(njack+1) + ijack + 1] += pairwt;
	}
	pc_log_log[ibin*(njack+1)] += pairwt;
	rp_log_log[ibin] += rp*pairwt;
	pi_log_log[ibin] += pi*pairwt;
      }
    }
  }
  return 0;
}

/* Minimumm variance weight */
double minVarWt(double s, double den)
{
  double ss, J3;
  if (J3_gamma <= 0.1) return 1.0;
  ss = s < J3_rmax ? s : J3_rmax;
  J3 = pow(J3_r0,J3_gamma) / (3-J3_gamma) * pow(ss,3-J3_gamma);
  return 1.0 / (1 + 4*M_PI*den*J3);
}
