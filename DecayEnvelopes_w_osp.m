p=input('Enter efficiency factor: ');
G=input('Enter the Gradient (Gauss/cm): ');
Gr=input('Enter the radial Gradient (Gauss/cm): ');
%T2=input('Enter T2: ');
osp=input('Enter the oversampling factor: ');

p_sw=p;
p_std=0.4;
Gamma=1126.;
dx=input('Enter the desired resolution (cm): ');

Km=osp*1./2./dx;
fprintf('Nominal radius in k-space: %f\n',Km/osp);
fprintf('Maximum radius in k-space: %f\n',Km);
fprintf('Nominal Read Time: %fms\n',Km/Gamma/Gr/osp*1000);

c1=0.54;
c2=0.46;
eta=pi/Km^3;
T2f=0.009;
T2s=0.027;

% T2f=1; % For checking Limiting forms of the PSF (i.e., no blurring)
% T2s=1;

DSW = @(x) (c1/exp(eta*x*x*x)+c2);

npts=4096;
k=linspace(0,Km,npts);

h=hann(2*npts);
w=ones(1,npts);
%win=w.*h(npts+1:end)';
 win=w; % For checking Limiting forms of the PSF (i.e., no blurring)

t_tpi=linspace(0,Km,npts);
t_tpi_std=linspace(0,Km,npts);
t_tpi_sw=linspace(0,Km,npts);

t_tpi=t_tpi/Gamma/G;
t_tpi_std=t_tpi_std/Gamma/Gr;
t_tpi_sw=t_tpi_sw/Gamma/G;

t0=p*Km/Gamma/G/osp;
t0_std=p_std*Km/Gamma/Gr/osp;
t0_sw=p_sw*Km/Gamma/G/osp;

index=floor(p*npts/osp);
t0=t_tpi(index);

index_std=floor(p_std*npts/osp);
t0_std=t_tpi_std(index_std);

index_sw=floor(p_sw*npts/osp);
t0_sw=t_tpi_sw(index_sw);

pKm3=p*Km*p*Km*p*Km/osp/osp/osp;
atpi=1/(3*G*Gamma*p*p*Km*Km/osp/osp);
for i=index+1:npts
    t_tpi(i)=(k(i)*k(i)*k(i)-pKm3)*atpi+t0;
end


pKm3_std=p_std*Km*p_std*Km*p_std*Km/osp/osp/osp;
atpi_std=1/(3*Gr*Gamma*p_std*p_std*Km*Km/osp/osp);
for i=index_std+1:npts
    t_tpi_std(i)=(k(i)*k(i)*k(i)-pKm3_std)*atpi_std+t0_std;
end

pKm3_sw=p_sw*Km*p_std*Km*p_sw*Km/osp/osp/osp;
atpi_sw=DSW(p_sw*Km/osp)*G*Gamma*p_sw*Km/osp*p_sw*Km/osp;
beta_sw=c2*pKm3_sw-c1/eta/exp(eta*pKm3_sw);
for i=index_sw+1:npts
    k3=k(i)*k(i)*k(i);
    t_tpi_sw(i)=(c2*k3-c1/eta/exp(eta*k3)-beta_sw)/3/atpi_sw+t0_sw;
end


fprintf('\nFinal Read Times-> TPI:%fms  SWTPI: %fms\n',t_tpi(npts)*1000,t_tpi_sw(npts)*1000);

arad=1./(Gr*Gamma);

t_rad=(arad*k);

% tpi_env=exp(-t_tpi/T2);
% 
% tpi_sw_env=exp(-t_tpi_sw/T2);
% 
% tpi_env_std=exp(-t_tpi_std/T2);

tpi_env=0.6*exp(-t_tpi/T2f)+0.4*exp(-t_tpi/T2s);

tpi_sw_env=0.6*exp(-t_tpi_sw/T2f)+0.4*exp(-t_tpi_sw/T2s);

tpi_env_std=0.6*exp(-t_tpi_std/T2f)+0.4*exp(-t_tpi_std/T2s);


rad_env=0.6*exp(-t_rad/T2f)+0.4*exp(-t_rad/T2s);

ratio=tpi_env./tpi_env_std;

density=sqrt(G/Gr);

snr=ratio/density;

filtered_tpi=win.*tpi_env;

filtered_tpi_sw=win.*tpi_sw_env;

sgm=linspace(0.01,4*Km,npts);
y=zeros(1,npts);
for i=1:npts
y(i)=norm((gaussmf(k, [sgm(i) 0])-filtered_tpi_sw),2)/norm(filtered_tpi_sw,2);
end
[m,ii]=min(y);
fprintf('\nBest Approximating Gaussian in k-space has sigma=%f/cm\n',sgm(ii));
fprintf('PSF for FT of this Gaussian has Predicted FWHM=%fcm\n\n',2.355/sgm(ii)/2/pi);

subplot(5,1,1);

plot(k,tpi_env,k,rad_env,k,tpi_env_std,k,filtered_tpi,k,filtered_tpi_sw);
legend('tpi envelope','radial envelope','tpi standard envelope', 'tpi envelope with han window');

subplot(5,1,2);

plot(k,ratio,k,snr);
legend('ratio of envelopes','snr');

subplot(5,1,3);

plot (k,gaussmf(k,[sgm(ii) 0]),k,filtered_tpi);
legend('Approximating Gaussian','Total k-space Envelope');
xlabel('K(1/cm)');

subplot(5,1,4);

plot (k,gaussmf(k,[sgm(ii) 0]),k,filtered_tpi_sw,'LineWidth',2);
legend('Approximating Gaussian','Total SW k-space Envelope');
xlabel('K(1/cm)');

nfft=32*npts;
g1=gaussmf(k,[sgm(ii),0]);
g2=g1(end:-1:2);
gg=cat(2,g2,g1);
gg_psf=abs((fftshift(fft(gg,nfft),2)));

hr=filtered_tpi_sw(end:-1:2);
hr2=cat(2,hr,filtered_tpi_sw);
hr2_psf=abs((fftshift(fft(hr2,nfft),2)));

dk=k(2)-k(1);
FOV=1./dk;
DX=1./2./Km;

% Note below that Matlab uses units of f/2*pi 
%xcord=linspace(-FOV/2*2*pi,FOV/2*2*pi,nfft+1);  % So that we have x=0 in the same position as the fft
xcord=linspace(-FOV/2,FOV/2,nfft+1);  % So that we have x=0 in the same position as the fft
xpos=xcord(1:nfft);
max_gg=max(gg_psf);
max_hr2=max(hr2_psf);

gg_psf=gg_psf/max_gg;
hr2_psf=hr2_psf/max_hr2;

nmin=nfft/2-160;
nmax=nfft/2+160;

subplot(5,1,5);
plot(xpos(nmin:nmax),gg_psf(nmin:nmax),xpos(nmin:nmax),hr2_psf(nmin:nmax),'LineWidth',2);
legend('Approximating Gaussian PSF','SW TPI PSF');
xlabel('X(cm)');

nrmse=norm((gaussmf(k, [sgm(ii) 0])-filtered_tpi_sw),2)/norm(filtered_tpi_sw,2);
fprintf('Decay Envelope NRMSE (k-space)=%f\n\n',nrmse);

x_NRMSE=norm((gg_psf-hr2_psf),2)/norm(hr2_psf,2);
fprintf('PSF NRMSE (x-space)=%f\n\n',x_NRMSE);

resol=zeros(1,nfft);
for i=1:nfft
resol(i)=abs(gg_psf(i)-0.5);
end


[rmin,irmin]=min(resol);
fprintf('Measured Gaussian PSF Width is=%fcm\n\n',abs(2*xpos(irmin))); %First will find the negative side
fprintf('Measured Sigma is=%fcm\n\n',abs(2*xpos(irmin)/2.355)); %First will find the negative side


resol=zeros(1,nfft);
for i=1:nfft
resol(i)=abs(hr2_psf(i)-0.5);
end

[rmin,irmin]=min(resol);
fprintf('Measured T2-blur PSF Width is=%fcm\n\n',abs(2*xpos(irmin))); %First will find the negative side

