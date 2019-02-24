
rm(list = ls())
  
set.seed(123)

X=rnorm(1000,0,1)

par(par(mfrow = c(1,1)))

plot(density(X, kernel="gaussian"), main="Densité Gaussienne")
plot(density(X, kernel="rectangular"), main="Densité Rectangulaire")
plot(density(X, kernel="triangular"), main="Densité Triangulaire")

require(MASS)

data(faithful)



X=faithful
h=seq(0.025,1,by=0.025) 
MISE=rep(0,length(h))

a=min(X$waiting)
b=max(X$waiting)

for (k in length(h)){
  est=density(X$waiting,bw=h[k],from=a, to=b, n=100)
  MISE[k]=sum((dnorm(est$x)-est$y)^2)/(b-a)/100
}

h_oracle=h[which.min(MISE)]
MISE_oracle=MISE[which.min(MISE)]
plot(density(X$waiting, bw=h_oracle, from=a, to=b, n=100), main="Densités non paramétriques")

est_ucv=density(X$waiting, bw="ucv")

plot(est_ucv)

est_ucv=density(X$eruptions, bw="ucv")
est_bcv=density(X$eruptions, bw="bcv")
est_sj=density(X$eruptions, bw="sj")

curve(est_bcv, add=TRUE, col="red")

MISE_ucv=sum((dnorm(est_ucv$x)-est$y)^2)/(b-a)/100
MISE_bcv=sum((dnorm(est_bcv$x)-est$y)^2)/(b-a)/100
MISE_sj=sum((dnorm(est_bcv$x)-est$y)^2)/(b-a)/100

h_ucv_oracle=h[which.min(MISE_bcv)]

MISE_ucv_oracle=MISE[which.min(MISE_ucv)]
MISE_bcv_oracle=MISE[which.min(MISE_bcv)]
MISE_sj_oracle=MISE[which.min(MISE_sj)]


plot(density(X$eruptions, bw="ucv"), main="Densités non paramétriques")

install.packages("cowplot")



#Superposition des graphes 

plot(est_ucv, ylim = c(0, 6)),
     xlim = range(est_ucv$x, est_bcv$x, est_sj$x), col = "blue", main="estimations non paramétriques selon les noyaux")

lines(est_bcv$x, est_bcv$y, col = "red")

lines(est_sj$x, est_sj$y, col = "yellow")

#legend("topleft", inset = 0.01, c("Noyau ucv", "Noyau bcv", "Noyaux SJ"),
       #col = c("blue", "red", "yellow"), lwd = 1)

n=200
sigma=0.05
epsilon=sigma*rnorm(n,0,1)
X=rnorm(n,0,1)
#mX=X^4
#mX=exp(X)
mX=X^2*exp(-3*abs(X))
Y=mX+epsilon
plot(X,mX)



#### Estimateur à noyau
a=min(X)
b=max(X)
grid=seq(a,b,by=(b-a)/500)
est_grid=rep(0,length(grid))
YK=matrix(rep(0,n*length(grid)),nrow=n)
XK=matrix(rep(0,n*length(grid)),nrow=n)
h=0.3
for(k in 1:length(grid)){
  for(i in 1:n){YK[i,k]=Y[i]*dnorm((grid[k]-X[i])/h);
  XK[i,k]=dnorm((grid[k]-X[i])/h)}}
est_grid=apply(YK,2,sum)/apply(XK,2,sum)
est_grid

plot(Y~X,type="p")
points(grid,est_grid,type="l",col="blue")
curve(x^2*exp(-3*abs(x)),type="l",col="red",add=TRUE )



nobs=1000;
X=rnorm(nobs,0,1)

plot(density(X,kernel="gaussian"))
curve(dnorm(x),add=TRUE)







