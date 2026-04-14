#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:51:10 2026

@author: sk322
"""

import numpy as np


# PYTHON TRANSLATION OF MF-ICA FUNCTIONS WRITTEN ORIGINALLY BY OLE WINTHER (2002, 2005)

#function [S,A,loglikelihood,Sigma,chi]=icaMF_jta_fixc_v3(X,par);

def icaMF_fixComp(X, par):
    
    A, sigma, par = initializeMF(x, par) # Initialise the MF-ICA code
    
    print(" Mean Field ICA empirical Bayes algorithm\n")
    print(" {} mean field solver, {} source prior\n".format(par['solver'],par['Sprior']))
    
    print(' {} parameter optimizer for {} A and {} noise\n'.format(par['optimizer'],par['Aprior'],par['Sigmaprior']))
    print(' sensors: D = {}\n sources: M = {}\n samples: N = {}\n\n'.format(par['D'],par['M'],par['N'])) 

    if par['draw']:
        if sigma.shape[0] * sigma.shape[1] == par['D']:
            print(' initial noise variance = {} \n\n'.format(np.sum(sigma)/par['D']))
        else:   
          print(' initial noise variance = {} \n\n'.format(np.trace(sigma)/sigma.shape[0]))


    # run mean field ICA empirical Bayes algorithm
    # This is a minimal working code. Hence I have stuck to AEM only
    
    if par['optimiser'] == 'aem':
        A, sigma = icaMFaem(A,sigma,X,par) # adaptive overrelaxed em by R Salakhutdinov + S Roweis
    else:
        print("The chosen optimiser hasn't been implemented yet. Try again.")
        
    # solve mean field equations and exit.
    theta, J, par = XASigma2thetaJ(X,A,Sigma,par)
    S,chi,f,ite,dSdS = par['mf_solver(theta,J,par)'] #===> WHat is this?
    
    loglikelihood = - f
    
    return S, A, loglikelihood, sigma, chi


# ================================================================== #
# ====== Parameter solvers ========================================= #
# ================================================================== #

# Define an EC solver
# Original by Ole Winther, IMM, February 2005
def ec_solver(theta, J, par):
    
    M = par['M']; N = par['N']
    
    try:
        minchi = par['minchi']
    except:
        minchi = 1e-7
    
    try:
        m = par['m']
    except:
        m = np.zeros(M,N)
        
    try:
        Sconst = par['Sconst']
    except:
        Sconst=np.zeros(M,1)
    
    # try:
    #     stitch = par['stitch']
    # except:
    #     stitch = zeros(M,1)
    # There is something buggy here. There is a "StitchSpectrum" code that doesn't actually exist in code. Hence omitting that case
    stitch = zeros(M,1)
        
    try:
        minLam_q = par['minLam_q']
    except: 
        minLam_q = 1e-5
        
    try:
        eta = par['eta']
    except:
        eta = 1
        
    dm = np.zeros(M,N); m_r = np.zeros(M,N); v_r = np.zeros(M,N)
    
    eigJ = eig(J); mineigJ = np.min(eigJ); maxeigJ = np.max(eigJ);
    Lam_r = 1 * (maxeigJ - mineigJ) * np.ones(M, 1)
    chi = np.linalg.inv(np.diag(Lam_r) - J)
    Lam_r = np.tile(Lam_r, (1, N))
    
    # set mean value of r-distribution to be the same as m
    gam_r = np.zeros(M,N)
    m_r = m
    v_r = np.diag(chi) # original comment by James T Allen: this line doesn't make sense, but seems to be overwritten later
    
    # make a covariance matrix for each sample
    chi = np.tile(chi, [1,1,N])
    
    ite = 0
    dmdm = np.inf
    dmdmN = np.inf*np.ones(N,1)
    tolN = par['S_tol'] / N
    I = np.arange(0, N)
    
    while (len(I)!=0 and (dmdm > par['S_tol']) and (ite < par['S_max_ite'])):
        ite += 1
        indx = np.random.randint(M)
        for sindx in range(M):
            cindx = indx[sindx]
            par[Sindx] = cindx
            
            # Find mean and variance of r
            m_r[cindx, I] = np.sum(np.multiply(np.moveaxis(chi[cindx,:,I], 0, -1), 
                                               (gam_r[:,I] + theta[:,I])), axis=1)
            v_r[cindx, I] = chi[cindx, cindx, I]
            
            # Find lagrange parameters of S
            Lam_s1 = 1  / np.max(minchi, v_r[cindx,I])
            gam_s1 = np.multiply(Lam_s1, m_r[cindx,I])
            
            # Update Lam_q
            gam_q = gam_s1 - gam_r[cindx, I]
            Lam_q = Lam_s1 - Lam_r[cindx, I]
            
            if Sconst[cindx] == 0:
                m[cindx,I],v_q = par['Smeanf']( gam_q , np.max( Lam_q , minLam_q ) , par ) # OBS Hack Here!!!
            elif Sconst[cindx] == 1:
                m[cindx,I] = par['m'][cindx,I]
                v_q = np.zeros(np.size(cindx), np.size(I))
                
            dm[cindx, I] = m[cindx, I] - m_r[cindx,I]
            
            # Find Lagrange parameters of s
            Lam_s2 = 1 / np.max(minchi, v_q)
            gam_s2 = np.multiply(Lam_2, m[cindx, I])
            
            # Update lam_r
            dgam_r = eta * ( gam_s2 - gam_s1 ) ; gam_r[cindx,I] = dgam_r + gam_r[cindx,I]
            dLam_r = eta * ( Lam_s2 - Lam_s1 ) ; Lam_r[cindx,I] = dLam_r + Lam_r[cindx,I]
            
            # update chi using Sherman-Woodbury
            if par['ecchiupdate'] == 'parallel':
                oM = np.ones(M, 1)
                kappa = dLam_r / (1+np.multiply(dLam_r, v_r[cindx, I]))
                kappa = np.reshape(kappa, (1, 1, len(I)))
                chiv = chi[:,cindx,I]; chiv = np.multiply(kappa[oM, :, :] , chiv); chiv = chiv[:,oM,:]
                chivt = chi[cindx,:,I]; chivt = chivt[oM,:,:]
                chi[:,:,I] = chi[:,:,I] - np.multiply(chiv, chivt)
            elif par['ecchiupdate'] == 'sequential':
                for j in range(len(I)):
                    i = I[j]
                    chiv = chi[:,cindx,i]
                    chi[:,:,i] = chi[:,:,i] - dLam_r[j] / ( 1 + dLam_r[j] * v_r[cindx,i] ) * chiv * chiv.T
 
        dmdmN=np.sum(np.multiply(dm, dm), 1)
        I=np.where(dmdmN > tolN)[0]    
        dmdm=np.sum(dmdmN)
        
    if nargout > 2:
        
        Lam_s = 1 / np.max(minchi, v_r); gam_s = np.multiply(Lam_s, m_r)
        gam_q = gam_s - gam_r; Lam_q = Lam_s - Lam_r
        
        par['Sindx'] = np.arange(M)
        par['Sindx'] = par['Sindx'][:, None] # IS THIS CORRECT?
        logZ_q = par['logZ0f'](gam_q, np.max( minLam_q , Lam_q ) , par)
        logZ_r = 0.5 * ( N * np.log(2*np.pi) - N * par['logdet2piSigma'] - par['XinvSigmaX'] ) ;
        for i in range(N):
            big_term = (gam_r[:,i] + theta[:,i]).T @ chi[:,:,i] @ (gam_r[:,i] + theta[:,i])
            logZ_r = logZ_r + 0.5 * np.linalg.slogdet(chi[:,:,i])[1] + 0.5*big_term
        logZ_s = 0.5 * ( N * np.log(2*np.pi) + np.sum(-np.log( Lam_s ) + (gam_s**2) / Lam_s ) )  
        G = ( - logZ_q - logZ_r + logZ_s ) / N

    
    return m,chi,G,ite,dmdm


# ======================================================================== #
# ====== Variational Solver ============================================== #
# ======================================================================== #

def variational_solver(theta, J, par):
    
    M = par['M']
    N = par['N']
    chi = np.zeros(M, M, N)
    
    # Initialise constants
    try:
        minichi = par['minichi']
    except:
        minichi = 1e-7
    try:
        m = par['m']
    except:
        m = np.zeros(M, N)
        
    dm = m
    Lam_q = - np.diag(J) # Variational result
    
    ite = 0
    dmdm = np.inf
    dmdmN = np.zeros(N,1)
    tolN = par['S_tol'] / N
    I = np.arange(N)
    
    while ((len(I) != 0) and (dmdm > par['S_tol']) and (ite < par['S_max_ite'])):
        
        ite += 1
        indx = np.random.randint(M)
        
        for sindx in range(M):
            
            cindx = indx[sindx]; par['Sindx'] = cindx # Current index
            
            # update hcav
            gam_q = theta[cindx, I] + (J[cindx, :] @ m[:,I]) + (Lam_q[cindx] * m[cindx, I])
            
            # Update m and derivative
            m_old = m[cindx, I]
            m[cindx, I] = par['Smeanf'](gam_q, Lam_q[cindx, np.ones(len(I))], par)
            dm[cindx, I] = m[cindx, I] - m_old
            
        dmdmN = np.sum(dm * dm, axis=1)
        I = np.where(dmdmN > tolN)[0]
        dmdm = np.sum(dmdmN)
        
    if nargout > 1: # Calculate chi
        gam_q = theta + ( J + np.diag(Lam_q) ) @ m
        par['Sindx'] = np.arange(M)
        par['Sindx'] = par['Sindx'][:, None] # IS THIS CORRECT?
        m_q, v_q = par['Smeanf'](gam_q, Lam_q[:, np.ones(N, 1)], par)
    
        for i in range(N):
            V_diag = np.diag(v_q[:, i])
            inv_term = np.eye(M) - (V_diag @ ( J + np.diag(Lam_q)))
            chi[:, :, i] = np.linalg.solve(inv_term, V_diag)
    
    if nargout > 2: # Calculate free energy per sample
    
        Lam_s = 1 / np.max(minchi, v_q); gam_s = np.multiply(Lasm_s, m_q)
        
        gam_r = gam_s - gam_q
        
        logZ_q = par['logZ0f'](gam_q, Lam_q[:,np.ones(N, 1)], par)
        logZ_r = 0.5 * (N * np.log(2*np.pi) - N*par['logdet2piSigma'] - par['XinvSigmaX'])
        for i in range(N):
            big_term = (gam_r[:,i] + theta[:,i]).T @ chi[:,:,i] @ (gam_r[:,i] + theta[:,i])
            logZ_r = logZ_r + 0.5 * np.linalg.slogdet(chi[:,:,i])[1] + 0.5*big_term

        logZ_s = 0.5 * ( N * np.log(2*np.pi) + np.sum(-np.log( Lam_s ) + (gam_s**2) / Lam_s ) )  
        G = ( - logZ_q - logZ_r + logZ_s ) / N
        
    
    return m,chi,G,ite,dmdm



# ================================================================== #
# ======== Helper functions ======================================== #
# ================================================================== #

# function [A,Sigma] = icaMFaem(A,Sigma,X,par) % aem

def icaMFaem(A, sigma, X, par):

    EM_step=0
    fold = np.inf
    alpha = 1.05 # increase factor for overrelaxed
    eta = 1 / alpha
    
    #if par['draw']
    #    h = waitbar(0,'Mean Field ICA - running AEM optimization...')
    
    while EM_step < par['max_ite']:
        
        EM_step += 1
        
        theta, J, par = XASigma2thetaJ(X,A, sigma, par)
        
        S, chi, f, S_ite, dSdS = par.mf_solver
    
    while EM_step<par.max_ite %& (dA_rel>tol | dSigma_rel>tol) 
    
        EM_step=EM_step+1;
        
        if par.draw waitbar(EM_step/par.max_ite,h), end
        
        [theta,J,par] = XASigma2thetaJ(X,A,Sigma,par) ;
        
        [S,chi,f,S_ite,dSdS] = par.mf_solver(theta,J,par) ;
        
    %    for i=1:par.M; subplot(par.M,1,i); plot(S(i,:)); end;
        
        [Aem,Sigmaem] = dASigmaf(X,A,Sigma,S,chi,par) ;
       
        if f > fold & eta > 1
            % we took a step that decreased the likelihood
            % backtrack and take regular em step instead
            eta = 1;
            A = Aold ; Sigma = Sigmaold ; f = fold ;
            Aem = Aemold ; Sigmaem = Sigmaemold ;
        else
            % increase step size
            eta = alpha * eta ;
            Aemold = Aem ; Sigmaemold = Sigmaem ;
        end 
      
        % aem update
        Aold = A ; Sigmaold = Sigma ;  fold = f ; 
        switch par.Aprior
            case 'free'
                A = A + eta * ( Aem - A ) ;
            case 'positive'
                A = A .* ( Aem ./ A ).^eta ;
                maxratio = 2; minratio = 0.5 ; % maximim increase/decrease factors
                A = min(A,maxratio*Aold); A = max(A,minratio*Aold);
        end 
          
        switch par.Sigmaprior 
            case 'free'
                logSigma = logm(Sigma) ; 
                Sigma = expm( logSigma + eta * ( logm(Sigmaem) - logSigma) ) ;
            case {'isotropic','diagonal'}
                Sigma = Sigma .* ( Sigmaem ./ Sigma ).^eta ;
        end
      
        dpoptrel = sum(abs([ (A(:) - Aold(:))'  (Sigma(:) - Sigmaold(:))' ])) / ...
                    ( sum(abs([Aold(:)' Sigmaold(:)'])) + eps ) ;
        
        if par.draw
            fprintf(' Iteration %d\n S_ite = %d   dSdS = %g  eta  = %g  dpar/|par| = %g\n',...
                EM_step,S_ite,dSdS,eta,dpoptrel);
            fprintf(' loglikelihood = %6.3f  ',-f);
            if size(Sigma,1)*size(Sigma,2)==par.D 
                fprintf(' noise variance = %g \n\n',sum(Sigma)/par.D);
            else       
                fprintf(' noise variance = %g \n\n',trace(Sigma)/size(Sigma,1));
            end
        end
    
    end % EM_step
    
    if par.draw close(h), end :
        
#============================================================================#
#========== Parameter Conversion ============================================#
#============================================================================#

def XASigma2thetaJ(X, A, Sigma, par):
    
    # invert Sigma
    if (np.shape(Sigma)[0] * np.shape(Sigma)[1]) == 1:
        invSigma = 1 / Sigma
        par['logdet2piSigma'] = par['D'] * np.log(2 * np.pi * Sigma)
    elif (np.shape(Sigma)[0] * np.shape(Sigma)[1]) == par['D']:
        invSigma = np.diag(1/Sigma)
        par['logdet2piSigma'] = np.sum(np.log(2*np.pi* Sigma))
    elif (np.shape(Sigma)[0] * np.shape(Sigma)[1]) == par['D']**2:
        invSigma = np.linalg.inv(Sigma)
        par['logdet2piSigma'] = np.linalg.slogdet(2 * np.pi * Sigma)[1]
        
    # Calculate external field and coupling matrix
    theta = A.T @ invSigma @ X
    J = -A.T @ invSigma @ A
    
    par['XinvSigmaX'] = np.sum(X * (invSigma @ X))
    
    return theta, J, par


def popt2ASigma(popt, par):
    
    D = par['D']; M = par['M']
    
    if par['Aprior']  == 'free':
        A = np.reshape(popt[0:par['Asize']], (D,M))
    elif par['Aprior'] == 'positive':
        A = np.exp(np.reshape(popt[0:par['Asize']], (D, M)))
    elif par['Aprior'] == 'constant':
        A = par['A_init']
        
    if par['Sigmaprior'] == 'free':
        par['sSigma'] = np.reshape(popt[par['Asize']+1:])
        
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                        parameter conversion                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [A,Sigma] = popt2ASigma(popt,par) ;

D = par.D ; M = par.M ;  

switch par.Aprior
    case 'free'
        A = reshape( popt(1:par.Asize) , D , M ) ;
    case 'positive'
        A = exp( reshape(popt(1:par.Asize), D , M ) ) ;
    case 'constant'
        A = par.A_init ;
end

switch par.Sigmaprior 
    case 'free' 
        par.sSigma = reshape(popt(par.Asize+1:end),par.D,par.D) ;
        Sigma = par.sSigma' * par.sSigma ;
    case {'isotropic','diagonal'}
        Sigma = exp( popt(par.Asize+1:end) ) ;
    case 'constant'
        Sigma = par.Sigma_init ;
end

function [popt] = ASigma2popt(A,Sigma,par) ;

D = par.D ; 

popt = zeros( par.Asize + par.Sigmasize , 1 ) ;

if strcmp(par.Aprior,'positive') & ...
        ( strcmp(par.optimizer,'bfgs') | strcmp(par.optimizer,'conjgrad') )
    popt(1:par.Asize) = log(A(:)) ;
else
    popt(1:par.Asize) = A(:) ;
end
 
switch par.Sigmaprior 
    case 'free' 
        popt(par.Asize+1:end) = par.sSigma(:) ; %R(par.triui) ;
    case {'isotropic','diagonal'}
        popt(par.Asize+1:end) = log(Sigma(:)) ;
    case 'constant'
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     parameter derivatives                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dpopt = dpoptf(X,A,Sigma,S,chi,par)
    
M = par.M ;

dpopt = zeros( par.Asize + par.Sigmasize , 1 ) ; 

for i=1:M  % set up matrices for finding derivatives
    for j=i:M
        tracechi(i,j)=sum(chi(i,j,:));
        tracechi(j,i)=tracechi(i,j);
        traceSS(i,j)=sum(chi(i,j,:))+sum(S(i,:)'.*S(j,:)');
        traceSS(j,i)=traceSS(i,j);
    end;
end;

if size(Sigma) == [par.D,1] 
    invSigma = diag(1 ./ Sigma ) ;
else
    invSigma = inv(Sigma) ;
end

dA = invSigma * ( A * traceSS - X * S' ) / par.N ;

switch par.Aprior
    case 'free'
        dpopt(1:par.Asize) = dA(:) ;
    case 'positive'
        dpopt(1:par.Asize) = A(:) .* dA(:) ;
    case 'constant'
end


switch par.Sigmaprior
    case 'isotropic'
        dSigma = 0.5 * invSigma * ( par.D  - ...
            ( sum(sum((X-A*S).^2)) + ... 
            sum(sum((A*tracechi).*A)) ) * invSigma / par.N ) ;
        dpopt(par.Asize+1) = Sigma * dSigma ;
    case 'diagonal'
        invSigma = diag( invSigma ) ;
        dSigma = 0.5 * invSigma .* ( ones(par.D,1) - ...
            ( sum((X-A*S).^2,2) + ... 
            sum((A*tracechi).*A,2) ) .* invSigma / par.N ) ;
        dpopt(par.Asize+1:end) =  Sigma .* dSigma ; 
    case 'free' 
        dSigma = 0.5 * invSigma * ( eye(par.D) - ...
            ( (X-A*S)*(X-A*S)' + A*tracechi*A' ) * invSigma / par.N ) ;  
        dSigma = 2 * dSigma - diag(diag(dSigma)) ; % take into account Sigma is symmetric
        dSigma = 2 * par.sSigma * dSigma ;
        dpopt(par.Asize+1:end) = dSigma(:) ;
    case 'constant'
end

function [A,Sigma] = dASigmaf(X,A,Sigma,S,chi,par)
    
M = par.M ;

for i=1:M  % set up matrices for finding derivatives
    for j=i:M
        tracechi(i,j)=sum(chi(i,j,:));
        tracechi(j,i)=tracechi(i,j);
        traceSS(i,j)=sum(chi(i,j,:))+sum(S(i,:)'.*S(j,:)');
        traceSS(j,i)=traceSS(i,j);
    end;
end;

if size(Sigma) == [par.D,1] 
    invSigma = diag(1 ./ Sigma ) ;
else
    invSigma = inv(Sigma) ;
end

switch par.Aprior
    case 'free'
        A = X * S' * inv(traceSS) ;
    case 'positive' % iterate eq. (2.13) in Ref. [3]
        A = A_positive_aem(traceSS,X * S',invSigma,A) ;
    case 'constant'
        A = par.A_init ;
end   

switch par.Sigmaprior
    case 'isotropic'
        Sigma = ( sum(sum((X-A*S).^2)) + ... 
            sum(sum((A*tracechi).*A)) ) / ( par.N * par.D ) ;
    case 'diagonal'
        Sigma = ( sum((X-A*S).^2,2) + ... 
                    sum((A*tracechi).*A,2) ) / par.N ;
    case 'free' 
        Sigma = ( (X-A*S)*(X-A*S)' + A*tracechi*A' ) / par.N ;  
    case 'constant'
        Sigma = par.Sigma_init ;
end


function [A] = A_positive(traceSS,XSt,invSigma,A) 

Atol = 1e-6;
KT_max_ite=100; 
A=A+10^-3*(A<eps);
sizeA = size(A,1) * size(A,2) ;
invSigmaXSt = invSigma * XSt ; 
amp=invSigmaXSt./max(eps,invSigma*A*traceSS);
Aneg=any(any(amp<0));
KT_ite=0; Aerror = inf ;
while ~Aneg & KT_ite<KT_max_ite & Aerror > Atol
  KT_ite=KT_ite+1;
  %dA = A - A.*amp ; dAerror = sum(sum(dA)) / (size(A,1)*size(A,2))
  Aold = A ;
  A=A.*amp ; 
  amp=invSigmaXSt./max(eps,invSigma*A*traceSS);
  Aneg=any(any(amp<0));
  Aerror = sum(sum(abs(A-Aold))) / sizeA ;
end
if Aneg % use quadratic programming instead - doesn't work for Sigma full
  M=size(traceSS,1);
  D=size(XSt,1);
  options=optimset('Display','off','TolX',10^5*eps,'TolFun',10^5*eps);
  for i=1:D
    B=quadprog(traceSS,-XSt(i,:)',[],[],[],[],zeros(M,1),[],A(i,:)',options);
    A(i,:)=B';
  end 
end

function [A] = A_positive_aem(traceSS,XSt,invSigma,A) 

Atol = 1e-6;
KT_max_ite=100; 
A=A+10^-3*(A<eps);
%if min(min(A)) < eps; A(find(A < eps)) = 10^-3; end  % not sure if this is ok - jta 31/8/2010
sizeA = size(A,1) * size(A,2) ;
invSigmaXSt = invSigma * XSt ; 
amp=invSigmaXSt./(invSigma*A*traceSS);
if any(any(amp < 0))
    amp(find(amp < 0)) = 10^-3;  % not sure if this is ok - jta 31/8/2010
end
f = sum( sum( traceSS .* ( A' * invSigma * A ) ) ) ...
    - 2 * sum( sum( A .* ( invSigma * XSt ) ) ) ;
Aneg=any(any(amp<0));
KT_ite=0; Aerror = inf ;
alpha = 2; %1.1 ;
eta = 1 ;
%maxratio=4; minratio=0.25; % maximum increase/decrease factors
while ~Aneg & KT_ite<KT_max_ite & Aerror > Atol 
  KT_ite=KT_ite+1;
  Aold = A ;
  A=A.*amp.^eta;
  %A=min(A,maxratio*Aold); A=max(A,minratio*Aold);
  fnew = sum( sum( traceSS .* ( A' * invSigma * A ) ) ) ...
    - 2 * sum( sum( A .* ( invSigma * XSt ) ) ) ;
  if f < fnew 
      eta = 1 ;
      A = Aold .* amp ;
      %A=min(A,maxratio*Aold); A=max(A,minratio*Aold);
      f = sum( sum( traceSS .* ( A' * invSigma * A ) ) ) ...
        - 2 * sum( sum( A .* ( invSigma * XSt ) ) ) ;
  else
      eta = alpha * eta ;
      f = fnew ;
  end
  amp=invSigmaXSt./(invSigma*A*traceSS);
  if any(any(amp < 0))
      amp(find(amp < 0)) = 10^-3;  % not sure if this is ok - jta 31/8/2010
  end
  Aneg=any(any(amp<0));
  Aerror = sum(sum(abs(A-Aold))) / sizeA ;
end
%KT_ite
if Aneg % use quadratic programming instead - doesn't work for Sigma full
  M=size(traceSS,1);
  D=size(XSt,1);
  options=optimset('Display','off','TolX',10^5*eps,'TolFun',10^5*eps);
  for i=1:D
    B=quadprog(traceSS,-XSt(i,:)',[],[],[],[],zeros(M,1),[],A(i,:)',options);
    A(i,:)=B';
  end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     initialization                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,Sigma,par] = initializeMF(X,par) 

% function handle to mean field solver
try 
    par.mf_solver = str2func( sprintf('%s_solver',par.solver) ) ;
catch
    par.mf_solver = str2func('ec_solver') ;
end
       

% method
try 
  switch par.method
    case 'constant'
      par.Aprior='constant'; par.Sigmaprior='constant';
    case 'fa' % factor analysis
      par.Aprior='free'; par.Sprior='Gauss'; par.Sigmaprior='diagonal';
    case 'neg_kurtosis'
      par.Sprior='bigauss';
    case 'pos_kurtosis'
      par.Sprior='mog'; 
    case 'positive'
      par.Sprior='exponential'; par.Aprior='positive';
    case 'ppca' % probabilistic PCA
      par.Aprior='free'; par.Sprior='Gauss'; par.Sigmaprior='isotropic';
  end
end   

% number of inputs, examples
[ par.D , par.N ] = size(X) ;

% A and number of sources
try par.Aprior; catch par.Aprior = 'free' ; end
try par.M = par.sources; catch par.M = par.D ; end  % default square mixing 
try 
    A = par.A_init;
    [ par.D , par.M ] = size( A ); 
catch
    Ascale = 0.1 / sqrt(par.M) ;
    A = Ascale * randn( par.D , par.M ); % use small random matrix  
    if strcmp(par.Aprior, 'positive' ) A = A .* sign(A) / sqrt( par.M) ; end
end
if strcmp( par.Aprior ,'constant' )
    par.Asize = 0 ;
    try par.A_init; 
    catch 
        error('par.A_init should be defined when par.Aprior = ''constant'''); 
    end
else
    par.Asize = par.D * par.M ;
end

% the sources
try % so far the initial value of S is only used to initialize Sigma!!!
    S = par.S_init ; 
    if size(S,1) ~= par.M
        error('size(S,1) must equal the number of sources'); 
    end
catch
    S=zeros(par.M,par.N); 
end
try % function handle to mean functions
    par.Smeanf = str2func(par.Sprior) ;
    par.logZ0f = str2func( sprintf('logZ0_%s',par.Sprior) ) ;
catch
    par.Smeanf = str2func('mog') ;
    par.logZ0f = str2func('logZ0_mog');
end
% parameters for prior distribution should be set here in field
% par.S(i).xxx
try par.Sprior; catch par.Sprior='mog'; end
switch lower(par.Sprior)
    case 'combi'
        for i=1:par.sources % maximum number of different priors
            try 
                par.S(i).meanf = str2func( par.S(i).prior ) ;
                par.S(i).logZ0f = str2func( sprintf('logZ0_%s',par.S(i).prior) ) ;
                switch par.S(i).prior
                    case 'mog'
                        try par.S(i).pi; catch par.S(i).pi = []; end 
                        try par.S(i).mu; catch par.S(i).mu = []; end 
                        try par.S(i).Sigma; catch par.S(i).Sigma = []; end
                        if isempty(par.S(i).pi) par.S(i).pi = [ 0.5 0.5 ]'; end
                        if isempty(par.S(i).mu) par.S(i).mu = [ 0 0 ]'; end
                        if isempty(par.S(i).Sigma) par.S(i).Sigma = [ 1 0.01 ]'; end
                end
                % parameter values for other priors should be added here!
            catch 
                error('Sprior is combi, but not all sources are specified in par.S(i).prior') ;
            end
        end
    case 'mog'
        try par.S.pi; catch par.S(1).pi = [0.5 0.5]'; end % mixing proportions
        try par.S.mu; catch par.S(1).mu = [0 0]'; end % means
        try par.S.Sigma; catch par.S(1).Sigma = [1 0.01]'; end % means
        Ssources = length(par.S) ;
        par.S(1).K = length(par.S(1).pi) ;
        if Ssources ~= par.sources % use same parameters for all sources 
            for indx=2:par.sources
                par.S(indx) = par.S(1) ;
            end
        end
end

% Sigma
try par.Sigmaprior; catch par.Sigmaprior = 'isotropic' ; end
switch par.Sigmaprior
    case 'isotropic'
        par.Sigmasize = 1 ;   
    case 'diagonal'
        par.Sigmasize = par.D ;
    case 'free'
        par.Sigmasize = par.D^2 ; 
    case 'constant'
        par.Sigmasize = 0 ; 
        try par.Sigma_init; catch 
            error('par.Sigma_init should be defined when par.Sigmaprior = ''constant'''); 
        end
end
try
  Sigma=par.Sigma_init;  
catch                
  Sigmascale = 1 ;
  switch par.Sigmaprior
    case {'isotropic','constant'}
      Sigma = Sigmascale * sum(sum( ( X - A * S ).^2 ) ) / (par.D*par.N) ;   
    case 'diagonal'
      Sigma = Sigmascale * sum( ( X - A * S ).^2 , 2 ) / par.N ; 
    case 'free'
      Sigma = Sigmascale * ( X - A * S ) * ( X - A * S )' / par.N ;
  end
end

% optimizer
try par.optimizer; catch par.optimizer = 'aem' ; end
if par.Asize + par.Sigmasize == 0 % no parameters to be optimized 
    par.optimizer = 'constant';
end

% more init of parameters...
if strcmp(par.Sigmaprior,'free') & ... 
        ( strcmp(par.optimizer,'bfgs') | strcmp(par.optimizer,'conjgrad') )
 % use par.sSigma as variable for unconstrained optimization in bfgs and
 % conjgrad
      par.sSigma = sqrtm(Sigma) ;  
end

% run time output
try par.draw ; catch par.draw = 1 ; end

% number of iterations and error tolerance
try par.max_ite; catch par.max_ite = 50 ; end
try par.S_tol; catch par.S_tol = 1e-10 ; end 
try par.S_max_ite; catch par.S_max_ite = 100 ; end

% chi update specific for ec_solver
try 
    par.ecchiupdate ;
catch
    if par.M > 10
        par.ecchiupdate = 'sequential' ;
    else
        par.ecchiupdate = 'parallel' ;
    end
end
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       mean (S) functions                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,df] = bigauss(gamma,lambda,par)
sigma2=1; mu=1;
lambda=1./(1+lambda*sigma2);
f=tanh(mu*gamma.*lambda);
if nargout > 1 
    df=lambda.*(sigma2+mu^2.*lambda.*(1-f.^2));
end
f=lambda.*(sigma2*gamma+mu*f);

function [f,df] = binary(gamma,lambda,par)
f=tanh(gamma);
if nargout > 1
    df=1-f.^2;
end

function [f,df] = binary_01(gamma,lambda,par)
tmp=exp(0.5*lambda+gamma);
f=1./(1+exp(0.5*lambda-gamma));
if nargout > 1 
    df=f.*(1-f);
end

function [f,df] = combi(gamma,lambda,par)
f = zeros(size(gamma)) ; df = zeros(size(gamma)) ;
Sindx = par.Sindx ;
for i=1:length(Sindx)
    par.Sindx = Sindx(i) ; % to pass information about current source
    [f(i,:),df(i,:)] = par.S(Sindx(i)).meanf(gamma(i,:),lambda(i,:),par) ;
end

function [f,df] = exponential(gamma,lambda,par)
try eta=par.expeta; catch eta = 1.0; end;
erfclimit=-35;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xi=(gamma-eta)./sqrt(lambda);
cc=(xi>erfclimit);
%i=find(cc==0);
%if ~isempty(i)
%    cc, pause
%end
xi1=xi.*cc;
epfrac=exp(-(xi1.^2)/2)./(Phi(xi1)*sqrt(2*pi));
f=cc.*(xi1+epfrac)./sqrt(lambda);            % need to go to higher order to get asymptotics right
if nargout > 1
    df=cc.*(1./lambda+f.*(xi1./sqrt(lambda)-f)); % need to go to higher order to get asymptotics right -fix at some point!!!
end

function [f,df] = Gauss(gamma,lambda,par)
f=gamma./(1+lambda);
if nargout > 1
    df=1./(1+lambda);
end

function [f,df] = heavy_tail(gamma,lambda,par)
alpha=1; % if changed change also in logZ0_heavy_tail
f=gamma./lambda-alpha*gamma./(2*alpha*lambda+gamma.^2);
if nargout > 1
    df=1./lambda+alpha*(gamma.^2-2*alpha*lambda)./(2*alpha*lambda+gamma.^2).^2;
end

function [f,df] = heavy_tail_plus_delta(gamma,lambda,par)
alpha=1; % if changed change also in logZ0_heavy_tail_plus_delta
beta=0.3; % proporation delta if changed change also in logZ0_heavy_tail_plus_delta
Z0ht=exp(0.5*gamma.^2./lambda).*(1+gamma.^2./(2*alpha*lambda)).^(-0.5*alpha);
f=(1-beta)*(gamma./lambda-alpha*gamma./(2*alpha*lambda+gamma.^2))./...
  (beta./Z0ht+(1-beta));
if nargout > 1 
    df=(1-beta)./(beta./Z0ht+(1-beta)).*...
        (1./lambda+alpha*(gamma.^2-2*alpha*lambda)./(2*alpha*lambda+gamma.^2).^2+...
        (gamma./lambda-alpha*gamma./(2*alpha*lambda+gamma.^2)).^2)-f.^2;
end
    
function [f,df] = Laplace(gamma,lambda,par)
erfclimit=-25;
eta=1;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xip=(gamma-eta)./sqrt(lambda);
ccp=(xip>erfclimit);ccpc=not(ccp);
xip1=ccp.*xip;
xim=-(gamma+eta)./sqrt(lambda);
ccm=(xim>erfclimit);ccmc=not(ccm);
xim1=ccm.*xim;
Dp=exp(-(xip1.^2)/2)/sqrt(2*pi);
Dm=exp(-(xim1.^2)/2)/sqrt(2*pi);
kp=Phi(xip1).*Dm;  
km=Phi(xim1).*Dp; 
f=ccp.*ccm.*(xip.*kp-xim.*km)./(sqrt(lambda).*(kp+km))+(-ccpc.*xim+ccmc.*xip)./sqrt(lambda);
if nargout > 1 
    df=(ccp.*ccm.*(1+xim.*xip+Dp.*Dm.*(xip+xim)./(kp+km)+sqrt(lambda).*(xip.*km-xim.*kp)./(kp+km).*f)+ccpc+ccmc)./lambda;
end

function [f,df] = mog(gamma,lambda,par)
[M N ] = size(gamma);
oN = ones(N,1) ;
K = length(par.S(par.Sindx(1)).pi) ;
oK = ones(K,1) ; 
for i = 1:length(par.Sindx) % loop over sources
    cindx = par.Sindx(i) ;
    cgamma = gamma(i,:) ; clambda = lambda(i,:) ; % 1 * N
    logpi = log(par.S(cindx).pi) ; % K * N
    cmu = par.S(cindx).mu ;        % K * 1
    cSigma = par.S(cindx).Sigma ; % K * 1
    mu2dSigma = cmu.^2 ./ cSigma ; 
    musqrtSigma = cmu .* sqrt(cSigma) ; 
    opSigmalambda = 1 + cSigma * clambda ;
    resp = logpi(:,oN) - 0.5 * log( opSigmalambda ) ....
       + 0.5 * ( sqrt(cSigma) * cgamma + musqrtSigma(:,oN) ) .^2 ./ ...
       opSigmalambda - 0.5 * mu2dSigma(:,oN) ; % K * N
    maxresp = max( resp, [] , 1 ) ;
    resp = exp( resp - maxresp(oK,:) ) ;
    normalizer = sum( resp, 1 ) ;
    mconst = ( cgamma(oK,:) + cmu(:,oN) ) .* cSigma(:,oN) ./ opSigmalambda ;
    f(i,:) = sum(  mconst .* resp , 1 ) ./ normalizer ;
    if nargout > 1 
        df(i,:) = ...
            sum( ( cSigma(:,oN) ./ opSigmalambda + mconst.^2 ) .* resp , 1 ) ./ ...
            normalizer ;
    end
end
if nargout > 1 
    df = df - f.^2 ;
end

function [f,df] = flat_pos(gamma,lambda,par)
kappa_a = -gamma./sqrt(lambda);
Dkappa=exp(-0.5*kappa_a.^2)./sqrt(2*pi);
Phikappa=Phi(kappa_a);
f=(gamma./lambda)+Dkappa./(sqrt(lambda).*(1-Phikappa));
if nargout > 1
    df=(1./lambda).*(1+kappa_a.*Dkappa./(1-Phikappa)-(Dkappa./(1-Phikappa)).^2);
end

function [f,df] = flat_01(gamma,lambda,par)
minlambda = 1e-10;
lambda0 = max(lambda, minlambda);
kappa_a = -gamma./sqrt(lambda0);
kappa_b = sqrt(lambda0) + kappa_a;
Dkappa_a = exp(-0.5*kappa_a.^2)./sqrt(2*pi);
Dkappa_b = exp(-0.5*kappa_b.^2)./sqrt(2*pi);
Phikappa_a = Phi(kappa_a);
Phikappa_b = Phi(kappa_b);
f = gamma./lambda0 + (Dkappa_a - Dkappa_b) ./ (sqrt(lambda0).*max((Phikappa_b - Phikappa_a),minlambda));
if nargout > 1
    df = (1./lambda0).*(1 + (kappa_a.*Dkappa_a - kappa_b.*Dkappa_b)./max((Phikappa_b - Phikappa_a),minlambda) - ...
        ((Dkappa_a - Dkappa_b)./max((Phikappa_a - Phikappa_b),minlambda)).^2);
end

function [f,df] = Gausspos(gamma,lambda,par)
mu = 0.0;
sigma = 1.0;
kappa = (gamma + mu./(sigma.^2))./(lambda + 1./(sigma.^2));
Dkappa = exp(-0.5*kappa.^2)./sqrt(2*pi);
Phikappa = Phi(kappa);
f = (gamma + mu./(sigma.^2)) ./ (lambda + 1./(sigma.^2)) + ...
    Dkappa ./ (sqrt(lambda + 1./(sigma.^2)) .* Phikappa);
if nargout > 1
    df = (mu./(lambda.*(sigma.^2) + 1)) .* ...
        (1 - kappa.*Dkappa./Phikappa - (Dkappa./Phikappa).^2);
end

function [f,df] = combi_exp(gamma,lambda,par)
f = zeros(size(gamma)) ; df = zeros(size(gamma)) ;
Sindx = par.Sindx ;
for i=1:length(Sindx)
    par.expeta = par.S(Sindx(i)).expeta ; 
    [f(i,:),df(i,:)] = exponential(gamma(i,:),lambda(i,:),par) ;
end

function [f,df] = xexp(gamma,lambda,par)
try eta=par.expeta; catch eta = 1.0; end;
erfclimit=-20;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xi=(gamma-eta)./sqrt(lambda);
cc=(xi>erfclimit);
%i=find(cc==0);
%if ~isempty(i)
%    cc, pause
%end
xi1=xi.*cc;
Phixi = Phi(xi1);
Dxi = exp(-0.5 * xi1.^2) / sqrt(2*pi);
f=cc.*((xi1.^2 + 1).*Phixi + xi1.*Dxi)./(sqrt(lambda).*(xi1.*Phixi + Dxi));            % need to go to higher order to get asymptotics right
if nargout > 1
    df=cc.*(1./lambda).*(2 - ((xi1.^2+1).*Phixi+xi1.*Dxi).*Phixi./((xi1.*Phixi+Dxi).^2)); % need to go to higher order to get asymptotics right -fix at some point!!!
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       logZ0 functions                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [logZ0] = logZ0_bigauss(gamma,lambda,par)
sigma2=1; mu=1;
lambda=1./(1+lambda*sigma2);
logZ0terms=0.5*(log(lambda)-mu^2/sigma2+lambda.*(mu^2/sigma2+gamma.^2*sigma2))...
           -log(2)+abs(gamma*mu.*lambda)+log(1+exp(-2*abs(gamma*mu.*lambda))); % log(cosh(gamma*mu.*lambda))
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_binary(gamma,lambda,par)
logZ0terms=-0.5*lambda-log(2)+abs(gamma)+log(1+exp(-2*abs(gamma)));
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_binary_01(gamma,lambda,par)
gamma=0.5*(gamma-0.5*lambda);
logZ0terms=gamma-log(2)+abs(gamma)+log(1+exp(-2*abs(gamma)));
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_combi(gamma,lambda,par)
logZ0=0;
for i=1:par.M % assumes that run through all sources
    par.Sindx = i ; % to pass information about current source
    logZ0 = logZ0 + par.S(i).logZ0f(gamma(i,:),lambda(i,:),par) ;
end


function [logZ0] = logZ0_exponential(gamma,lambda,par)
try eta=par.expeta; catch eta = 1.0; end;
erfclimit=-35;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xi=(gamma-eta)./sqrt(lambda);
cc=(xi>erfclimit);
xi1=xi.*cc;
logZ0terms=cc.*(log(Phi(xi1))+0.5*log(2*pi)+0.5*xi1.^2)-(1-cc).*log(abs(xi)+cc)+log(eta)-0.5*log(lambda);
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_Gauss(gamma,lambda,par)
logZ0terms=0.5*gamma.^2./(1+lambda)-0.5*log(1+lambda);
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_heavy_tail(gamma,lambda,par)
alpha=1; % if changed change also in heavy_tail.m
logZ0terms=0.5*gamma.^2./lambda-0.5*alpha*log(1+gamma.^2./(2*lambda*alpha));
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_heavy_tail_plus_delta(gamma,lambda,par)
alpha=1; % if changed change also in heavy_tail_plus_delta
beta=0.3; % proporation delta if changed change also in heavy_tail_plus_delta
Z0ht=exp(0.5*gamma.^2./lambda).*(1+gamma.^2./(2*alpha*lambda)).^(-0.5*alpha);
logZ0terms=log(beta+(1-beta)*Z0ht);
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_Laplace(gamma,lambda,par)
erfclimit=-25;
eta=1;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xip=(gamma-eta)./sqrt(lambda);
ccp=(xip>erfclimit);ccpc=not(ccp);
xip1=ccp.*xip;
xim=-(gamma+eta)./sqrt(lambda);
ccm=(xim>erfclimit);ccmc=not(ccm);
xim1=ccm.*xim;
Dp=exp(-(xip1.^2)/2)/sqrt(2*pi);
Dm=exp(-(xim1.^2)/2)/sqrt(2*pi);
Phip=Phi(xip1);  
Phim=Phi(xim1); 
logZ0terms=log(0.5*eta)+0.5*log(2*pi./lambda)+... 
ccp.*ccm.*(0.5*xip1.^2+0.5*xim1.^2+log(Dm.*Phip+Dp.*Phim)-0.5*log(2*pi))+...
ccp.*ccmc.*(0.5*xip1.^2+log(Phip+Dp./(abs(xim)+ccm)))+...
ccpc.*ccm.*(0.5*xim1.^2+log(Phim+Dm./(abs(xip)+ccp)))+...
ccpc.*ccmc.*(log(1./(abs(xip)+ccp)+1./(abs(xim)+ccm))-0.5*log(2*pi));
logZ0=sum(sum(logZ0terms));

function [logZ0] = logZ0_mog(gamma,lambda,par)
oN = ones(size(gamma,2),1) ;
K = length(par.S(par.Sindx(1)).pi) ;
oK = ones(K,1) ;
logZ0 = 0 ;
for i = 1:length(par.Sindx) % loop over sources
    cindx = par.Sindx(i) ;
    cgamma = gamma(i,:) ; clambda = lambda(i,:) ; % 1 * N
    logpi = log(par.S(cindx).pi) ; % K * 1
    cmu = par.S(cindx).mu ;       % K * 1
    cSigma = par.S(cindx).Sigma ; % K * 1
    mu2dSigma = cmu .^ 2 ./ cSigma ; 
    musqrtSigma = cmu .* sqrt(cSigma) ; 
    resp = logpi(:,oN) - 0.5 * log( 1 + cSigma * clambda ) ....
       + 0.5 * (sqrt(cSigma) * cgamma + musqrtSigma(:,oN) ) .^2 ./ ...
       ( 1 + cSigma * clambda ) - 0.5 * mu2dSigma(:,oN) ; % K * N
    maxresp = max( resp, [] , 1 ) ;
    logZ0 = logZ0 + sum ( maxresp + log( sum( exp( resp - maxresp(oK,:) ) , 1 ) ) ) ; 
end

function [logZ0] = logZ0_flat_pos(gamma,lambda,par)
% this is clearly not correct
logZ0 = 1;

function [logZ0] = logZ0_flat_01(gamma,lambda,par)
% this is also clearly not correct
logZ0 = 1;

function [logZ0] = logZ0_Gausspos(gamma,lambda,par)
% this is again clearly not correct
logZ0 = 1;

function [logZ0] = logZ0_combi_exp(gamma,lambda,par)
logZ0=0;
for i=1:par.M % assumes that run through all sources
    par.expeta = par.S(i).expeta;
    logZ0 = logZ0 + logZ0_exponential(gamma(i,:),lambda(i,:),par) ;
end

function [logZ0] = logZ0_xexp(gamma,lambda,par)
% this is another incorrect function
logZ0 = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                different math help function                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Phi (error) function
function y=Phi(x)
% Phi(x) = int_-infty^x Dz 
z=abs(x/sqrt(2));
t=1.0./(1.0+0.5*z);
y=0.5*t.*exp(-z.*z-1.26551223+t.*(1.00002368+...
    t.*(0.37409196+t.*(0.09678418+t.*(-0.18628806+t.*(0.27886807+...
    t.*(-1.13520398+t.*(1.48851587+t.*(-0.82215223+t.*0.17087277)))))))));
y=(x<=0.0).*y+(x>0).*(1.0-y);



% ==========  auxiliary functions  =================================

function  [x,n, F,g, opts] = check(fun,par,x0,opts0)
% Check function call
  x = x0(:);   sx = size(x);   n = max(sx);   
  if  (min(sx) > 1)
      error('x0  should be a vector'), end
  [F g] = feval(fun,x,par);
  sf = size(F);   sg = size(g);
  if  any(sf-1) | ~isreal(F)
      error('F  should be a real valued scalar'), end
  if  (min(sg) ~= 1) | (max(sg) ~= n)
      error('g  should be a vector of the same length as  x'), end
  so = size(opts0);
  if  (min(so) ~= 1) | (max(so) < 4) | any(~isreal(opts0(1:4)))
      error('opts  should be a real valued vector of length 4'), end
  opts = opts0(1:4);   opts = opts(:).';
  i = find(opts <= 0);
  if  length(i)    % Set default values
    d = [1  1e-4*norm(g, inf)  1e-8  100];
    opts(i) = d(i);
  end   
% ----------  end of  check  ---------------------------------------

function  D = checkD(n,D0)
% Check given inverse Hessian
  D = D0;   sD = size(D);
  if  any(sD - n)
      error(sprintf('D  should be a square matrix of size %g',n)), end
  % Check symmetry
  dD = D - D';   ndD = norm(dD(:),inf);
  if  ndD > 10*eps*norm(D(:),inf)
      error('The given D0 is not symmetric'), end
  if  ndD,  D = (D + D')/2; end  % Symmetrize      
  [R p] = chol(D);
  if  p
      error('The given D0 is not positive definite'), end
    
function  [alpha,fn,gn,neval,slrat] = ...
              softline(fun,fpar, x,f,g, h)
% Soft line search:  Find  alpha = argmin_a{f(x+a*h)}
  % Default return values 
  alpha = 0;   fn = f;   gn = g;   neval = 0;  slrat = 1;
  n = length(x);  
  
  % Initial values
  dfi0 = dot(h,gn);  if  dfi0 >= 0,  return, end
  fi0 = f;    slope0 = .05*dfi0;   slopethr = .995*dfi0;
  dfia = dfi0;   stop = 0;   ok = 0;   neval = 0;   b = 1;
  
  while   ~stop
    [fib g] = feval(fun,x+b*h,fpar);  neval = neval + 1;
    dfib = dot(g,h); 
    if  b == 1, slrat = dfib/dfi0; end
    if  fib <= fi0 + slope0*b    % New lower bound
      if  dfib > abs(slopethr),  stop = 1;
      else
        alpha = b;   fn = fib;   gn = g;   dfia = dfib;  
        ok = 1;   slrat = dfib/dfi0;
        if  (neval < 5) & (b < 2) & (dfib < slopethr)
          % Augment right hand end
          b = 2*b;
        else,  stop = 1; end
      end
    else,  stop = 1; end   
  end
  
  stop = ok;  xfd = [alpha fn dfia; b fib dfib; b fib dfib];
  while   ~stop
    c = interpolate(xfd,n);
    [fic g] = feval(fun, x+c*h, fpar);   neval = neval+1;
    xfd(3,:) = [c  fic  dot(g,h)];
    if fic < fi0 + slope0*c    % New lower bound
      xfd(1,:) = xfd(3,:);   ok = 1;
      alpha = c;   fn = fic;   gn = g;  slrat = xfd(3,3)/dfi0;
    else,  xfd(2,:) = xfd(3,:);  ok = 0; end
    % Check stopping criteria
    ok = ok & abs(xfd(3,3)) <= abs(slopethr);
    stop = ok | neval >= 5 | diff(xfd(1:2,1)) <= 0;
  end  % while   
%------------  end of  softline  ------------------------------

#========== Function for interpolations ================#

def interpolate(xfd, n):

  
function  alpha = interpolate(xfd,n);
% Minimizer of parabola given by
% xfd(1:2,1:3) = [a fi(a) fi'(a); b fi(b) dummy]

  a = xfd(1,1);   b = xfd(2,1);   d = b - a;   dfia = xfd(1,3);
  C = diff(xfd(1:2,2)) - d*dfia;
  if C >= 5*n*eps*b    % Minimizer exists
    A = a - .5*dfia*(d^2/C);
    d = 0.1*d;   alpha = min(max(a+d, A), b-d);
  else
    alpha = (a+b)/2;
  end
%------------  end of  interpolate  --------------------------

