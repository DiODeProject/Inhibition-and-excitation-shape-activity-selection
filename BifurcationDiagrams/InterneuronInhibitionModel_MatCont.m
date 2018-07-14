function out = InterneuronInhibitionModel_MatCont
out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = @der3;
out{8} = [];
out{9} = [];

% --------------------------------------------------------------------------
function dydt = fun_eval(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
f1=1/(1+exp(-g1*(kmrgd(1)-b1)));
f2=1/(1+exp(-g1*(kmrgd(2)-b1)));
alpha1=beta*r *f1;
alpha2=beta*r *f2;
Inp1=q*(dm+deld/2);
Inp2=q*(dm-deld/2);
beta1=beta/(1+exp(-g2*(kmrgd(3)-b2)));
beta2=beta/(1+exp(-g2*(kmrgd(3)-b2)));
dydt=[-k*kmrgd(1)+alpha1-beta1+Inp1;
-k*kmrgd(2)+alpha2-beta2+Inp2;
-kinh*kmrgd(3)+w*(f1+f2);];

% --------------------------------------------------------------------------
function [tspan,y0,options] = init
handles = feval(InterneuronInhibitionModel_MatCont);
y0=[0,0,0];
options = odeset('Jacobian',handles(3),'JacobianP',handles(4),'Hessians',handles(5),'HessiansP',handles(6));
tspan = [0 10];

% --------------------------------------------------------------------------
function jac = jacobian(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
jac=[ (beta*g1*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - k , 0 , -(beta*g2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , (beta*g1*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - k , -(beta*g2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; (g1*w*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , (g1*w*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , -kinh ];
% --------------------------------------------------------------------------
function jacp = jacobianp(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
jacp=[ r/(exp(g1*(b1 - kmrgd(1))) + 1) - 1/(exp(g2*(b2 - kmrgd(3))) + 1) , beta/(exp(g1*(b1 - kmrgd(1))) + 1) , q , q/2 , -kmrgd(1) , 0 , 0 , deld/2 + dm , -(beta*r*exp(g1*(b1 - kmrgd(1)))*(b1 - kmrgd(1)))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , (beta*exp(g2*(b2 - kmrgd(3)))*(b2 - kmrgd(3)))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 , -(beta*g1*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , (beta*g2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; r/(exp(g1*(b1 - kmrgd(2))) + 1) - 1/(exp(g2*(b2 - kmrgd(3))) + 1) , beta/(exp(g1*(b1 - kmrgd(2))) + 1) , q , -q/2 , -kmrgd(2) , 0 , 0 , dm - deld/2 , -(beta*r*exp(g1*(b1 - kmrgd(2)))*(b1 - kmrgd(2)))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , (beta*exp(g2*(b2 - kmrgd(3)))*(b2 - kmrgd(3)))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 , -(beta*g1*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , (beta*g2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , 0 , 0 , 0 , 0 , -kmrgd(3) , 1/(exp(g1*(b1 - kmrgd(1))) + 1) + 1/(exp(g1*(b1 - kmrgd(2))) + 1) , 0 , -w*((exp(g1*(b1 - kmrgd(1)))*(b1 - kmrgd(1)))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 + (exp(g1*(b1 - kmrgd(2)))*(b1 - kmrgd(2)))/(exp(g1*(b1 - kmrgd(2))) + 1)^2) , 0 , -w*((g1*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 + (g1*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2) , 0 ];
% --------------------------------------------------------------------------
function hess = hessians(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
hess1=[ (2*beta*g1^2*r*exp(2*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 - (beta*g1^2*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , 0 , 0 ; 0 , 0 , 0 ; (2*g1^2*w*exp(2*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 - (g1^2*w*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , 0 , 0 ];
hess2=[ 0 , 0 , 0 ; 0 , (2*beta*g1^2*r*exp(2*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 - (beta*g1^2*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , 0 ; 0 , (2*g1^2*w*exp(2*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 - (g1^2*w*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , 0 ];
hess3=[ 0 , 0 , (beta*g2^2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 - (2*beta*g2^2*exp(2*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 ; 0 , 0 , (beta*g2^2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 - (2*beta*g2^2*exp(2*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 ; 0 , 0 , 0 ];
hess(:,:,1) =hess1;
hess(:,:,2) =hess2;
hess(:,:,3) =hess3;
% --------------------------------------------------------------------------
function hessp = hessiansp(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
hessp1=[ (g1*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , 0 , -(g2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , (g1*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , -(g2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , 0 , 0 ];
hessp2=[ (beta*g1*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , 0 , 0 ; 0 , (beta*g1*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , 0 ; 0 , 0 , 0 ];
hessp3=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
hessp4=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
hessp5=[ -1 , 0 , 0 ; 0 , -1 , 0 ; 0 , 0 , 0 ];
hessp6=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , -1 ];
hessp7=[ 0 , 0 , 0 ; 0 , 0 , 0 ; (g1*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 , (g1*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 , 0 ];
hessp8=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
hessp9=[ (beta*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 + (beta*g1*r*exp(g1*(b1 - kmrgd(1)))*(b1 - kmrgd(1)))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - (2*beta*g1*r*exp(2*g1*(b1 - kmrgd(1)))*(b1 - kmrgd(1)))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 , 0 , 0 ; 0 , (beta*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 + (beta*g1*r*exp(g1*(b1 - kmrgd(2)))*(b1 - kmrgd(2)))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - (2*beta*g1*r*exp(2*g1*(b1 - kmrgd(2)))*(b1 - kmrgd(2)))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 , 0 ; (w*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 + (g1*w*exp(g1*(b1 - kmrgd(1)))*(b1 - kmrgd(1)))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - (2*g1*w*exp(2*g1*(b1 - kmrgd(1)))*(b1 - kmrgd(1)))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 , (w*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 + (g1*w*exp(g1*(b1 - kmrgd(2)))*(b1 - kmrgd(2)))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - (2*g1*w*exp(2*g1*(b1 - kmrgd(2)))*(b1 - kmrgd(2)))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 , 0 ];
hessp10=[ 0 , 0 , (2*beta*g2*exp(2*g2*(b2 - kmrgd(3)))*(b2 - kmrgd(3)))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 - (beta*g2*exp(g2*(b2 - kmrgd(3)))*(b2 - kmrgd(3)))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 - (beta*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , 0 , (2*beta*g2*exp(2*g2*(b2 - kmrgd(3)))*(b2 - kmrgd(3)))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 - (beta*g2*exp(g2*(b2 - kmrgd(3)))*(b2 - kmrgd(3)))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 - (beta*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , 0 , 0 ];
hessp11=[ (beta*g1^2*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - (2*beta*g1^2*r*exp(2*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 , 0 , 0 ; 0 , (beta*g1^2*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - (2*beta*g1^2*r*exp(2*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 , 0 ; (g1^2*w*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - (2*g1^2*w*exp(2*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 , (g1^2*w*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - (2*g1^2*w*exp(2*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 , 0 ];
hessp12=[ 0 , 0 , (2*beta*g2^2*exp(2*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 - (beta*g2^2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , 0 , (2*beta*g2^2*exp(2*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 - (beta*g2^2*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 ; 0 , 0 , 0 ];
hessp(:,:,1) =hessp1;
hessp(:,:,2) =hessp2;
hessp(:,:,3) =hessp3;
hessp(:,:,4) =hessp4;
hessp(:,:,5) =hessp5;
hessp(:,:,6) =hessp6;
hessp(:,:,7) =hessp7;
hessp(:,:,8) =hessp8;
hessp(:,:,9) =hessp9;
hessp(:,:,10) =hessp10;
hessp(:,:,11) =hessp11;
hessp(:,:,12) =hessp12;
%---------------------------------------------------------------------------
function tens3  = der3(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
tens31=[ (beta*g1^3*r*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - (6*beta*g1^3*r*exp(2*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 + (6*beta*g1^3*r*exp(3*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^4 , 0 , 0 ; 0 , 0 , 0 ; (g1^3*w*exp(g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^2 - (6*g1^3*w*exp(2*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^3 + (6*g1^3*w*exp(3*g1*(b1 - kmrgd(1))))/(exp(g1*(b1 - kmrgd(1))) + 1)^4 , 0 , 0 ];
tens32=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
tens33=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
tens34=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
tens35=[ 0 , 0 , 0 ; 0 , (beta*g1^3*r*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - (6*beta*g1^3*r*exp(2*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 + (6*beta*g1^3*r*exp(3*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^4 , 0 ; 0 , (g1^3*w*exp(g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^2 - (6*g1^3*w*exp(2*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^3 + (6*g1^3*w*exp(3*g1*(b1 - kmrgd(2))))/(exp(g1*(b1 - kmrgd(2))) + 1)^4 , 0 ];
tens36=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
tens37=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
tens38=[ 0 , 0 , 0 ; 0 , 0 , 0 ; 0 , 0 , 0 ];
tens39=[ 0 , 0 , (6*beta*g2^3*exp(2*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 - (beta*g2^3*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 - (6*beta*g2^3*exp(3*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^4 ; 0 , 0 , (6*beta*g2^3*exp(2*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^3 - (beta*g2^3*exp(g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^2 - (6*beta*g2^3*exp(3*g2*(b2 - kmrgd(3))))/(exp(g2*(b2 - kmrgd(3))) + 1)^4 ; 0 , 0 , 0 ];
tens3(:,:,1,1) =tens31;
tens3(:,:,1,2) =tens32;
tens3(:,:,1,3) =tens33;
tens3(:,:,2,1) =tens34;
tens3(:,:,2,2) =tens35;
tens3(:,:,2,3) =tens36;
tens3(:,:,3,1) =tens37;
tens3(:,:,3,2) =tens38;
tens3(:,:,3,3) =tens39;
%---------------------------------------------------------------------------
function tens4  = der4(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
%---------------------------------------------------------------------------
function tens5  = der5(t,kmrgd,beta,r,dm,deld,k,kinh,w,q,g1,g2,b1,b2)
