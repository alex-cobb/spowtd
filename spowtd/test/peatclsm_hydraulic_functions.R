########################################################################
################# Load packages  #######################################
########################################################################

#Load necessary packages
library(stats)


# CLI argument for whether to print specific yield or transmissivity
args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1) {
    stop("Exactly one argument required", call.=FALSE)
}
what_to_print <- args[1]
if ((what_to_print != "specific_yield")
    &
    (what_to_print != "transmissivity")) {
    stop("Argument must be either 'specific_yield' or 'transmissivity'")
}


########################################################################
###### Specific yield with Campbell function and microtopography #######
########################################################################
### INPUT PREPARATION
# predefine matrix to store PEATCLSMTN hydraulic parameters
phydr <- as.data.frame(matrix(ncol = 4, nrow = 1))
colnames(phydr) <- c("theta_s","Ks","b","psi_s") 

# hydraulic parameter values of PEATCLSMTN (Table 1 in Apers et al. 2022, JAMES)
phydr$Ks = 6*10^(-5) *100*84000  # [m3.m-3]   # saturated hydraulic conductivity
phydr$theta_s = 0.88             # [m3.m-3]   # saturated moisture content
phydr$b = 7.4                    # [-]        # shape parameter
phydr$psi_s = -0.024             # [m]        # air entry pressure

# standard deviation of PEATCLSMTN microtopographic distribution
sd = 0.162    # [m]
  
# predefine variable sequences
zl_ = seq(-1,1,0.01)                    # lower water level in [m]
zu_ = seq(-0.99,1.01,0.01)              # upper water level in [m]
Sy_soil = as.numeric(seq_len(201))[NA]  # same length variable
wl=(zl_+zu_)/2                          # mean water level in [m]


########################################################################
### DEFINE FUNCTIONS
# function to calculate soil moisture profile based on Campbell function and microtopography (Equations 4/5 in Dettmann & Bechtold 2015, Hydrological Processes)
Campbell_1d_Az <- function(z_,zlu,theta_s,psi_s,b){
  Fs = pnorm(z_, mean = 0, sd, log = FALSE) # PEATCLSMTN microtopographic distribution with 0.16 m as standard deviation
  theta = theta_s * (((zlu-z_)*100)/(psi_s*100))^(-1/b)
  theta[((zlu-z_)*100)>=(psi_s*100)]=theta_s
  theta_Fs = (1-Fs)*theta
  return(theta_Fs)
}

# calculation of soil specific yield profiles (Equation 1 in Dettmann & Bechtold 2015, Hydrological Processes)
get_Sy_soil <- function(zl_,zu_,phydr){
  theta_s = phydr$theta_s
  b = phydr$b
  psi_s = phydr$psi_s
  for (i in 1:length(zl_)){
    zl=zl_[i]
    zu=zu_[i]
    A = 0
    for (j in 1:200){
      zm=0.5*(zl_[j]+zu_[j])
      Azl = Campbell_1d_Az(zm,zl,theta_s,psi_s,b) # apply campbell function to get soil moisture profile for lower (zl) water level
      Azu = Campbell_1d_Az(zm,zu,theta_s,psi_s,b) # apply campbell function to get soil moisture profile for upper (zu) water level
      A = A + (zu_[j]-zl_[j]) *  (Azu-Azl)
    }
    Sy_soil[i] = 1/(1*(zu-zl)) * A
  }
  return(Sy_soil)
}


########################################################################
### CALCULATE THE SPECIFIC YIELD (Dettmann and Bechtold 2015, Hydrological Processes)
Sy1_soil = get_Sy_soil(zl_,zu_,phydr)                                 # Function above refering to Equation 1 in Dettmann and Bechtold 2015
Sy1_surface = pnorm(0.5*(zu_+zl_), mean = 0, sd, log = FALSE)         # Equation 3 in Dettmann and Bechtold 2015 (accounting for the hummocks/hollows)
Sy1 = Sy1_soil + Sy1_surface                                          # Equation 2 in Dettmann and Bechtold 2015


#######################################################################################
### TABULATE SPECIFIC YIELD FUNCTION

if (what_to_print == "specific_yield") {
    write.csv(
        data.frame(
            water_level_m = wl,
            specific_yield = Sy1
        ),
        stdout(),
        row.names = FALSE
    )
}


########################################################################
###################### Discharge - WL relation #########################
########################################################################
### INPUT PREPARATION 
# parameter values of PEATCLSMTN (Table 1 in Apers et al. 2022, JAMES)
Ksmacz0 = 7.3           # [m/s]
alpha = 3               # [-]      
c = 1.5* 10^(-5)        # [m^(-1)] 

# predefine variable sequences
z = -seq(-0.0,1.5,0.01) # water level [m]


########################################################################
### DEFINE FUNCTIONS
# Define transmissivity function
Transmissivity <- function(Ksmacz0,alpha,z) {
  Transmissivity_S = (Ksmacz0*(1 - z*100)^(1-alpha))/(100*(alpha-1))
  return(Transmissivity_S)
}


######################################################################################
### CALCULATE THE DISCHARGE FUNCTION
Transmissivity_S = Transmissivity(Ksmacz0,alpha,z)    # Transmissivity (Equation 3 in Apers et al. 2022, JAMES) in [m2/s]
Discharge = Transmissivity_S * c                      # Discharge (Equation 4 in Apers et al. 2022, JAMES) 


#######################################################################################
### TABULATE DISCHARGE FUNCTION

if (what_to_print == 'transmissivity') {
    write.csv(
        data.frame(
            water_level_m = z,
            transmissivity_m2_s = Transmissivity_S
        ),
        stdout(),
        row.names = FALSE
    )
}
