#!/usr/bin/env python3
"""RL-Repair as a standalone Axelrod strategy.  The 85-feature MLP weights are embedded; only sample mode is supported.
Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
Use the module-level ``GAME`` with axelrod Match/Tournament.
"""
from __future__ import annotations

import base64
import json
import math
import random
import zlib
from operator import mul
from typing import Sequence

from axelrod import Action, Game, Player

# Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
GAME = Game(r=6, s=0, t=10, p=1)

# Internal action convention used by the original pri_dile policies.
C = 0
D = 1

# Axelrod action objects.
A_C = Action.C
A_D = Action.D


def _histories(player, opponent):
    mine = [D if move == A_D else C for move in player.history]
    theirs = [D if move == A_D else C for move in opponent.history]
    if len(mine) != len(theirs):
        raise RuntimeError("history lengths differ")
    return mine, theirs


def _require_sample(mode, name):
    if mode != "sample":
        raise ValueError(f"{name} supports only 'sample' mode")

_EMBEDDED = (
    "c-"
    "mCG>#k+Nk(~D~{7nk0?p1mfK@bRe1P>sOgwaTbpqF=ik(s+`85onz?sN8Dt12&%5t0ArFaPDg{p<hp_ka7#zyHfW|M&mvZ~yX7|M"
    "Oq}{$Kz0-~Zcx{pbJuzyJ2X+F$<qm;drlfB(1t`LF-"
    "<pZ?cB{p<hu=l}LE|Np<3|KtCNzxpr#ss8(4{{3Bl{mb9~^S}M`|N8sC#lQNuzyHhst-teM|Ng)AgZ}dG|HJ?J%m1tY`yal4*V>!"
    "=j`zLUc6;~y^>6%P8>g?f)_U#jdH(wSyRV<wPyhP&zODcB7Ju^_eSdqbw@&Nr&d1+8bN0K>S?k_+-{8;I?R&lcoqp$B-"
    "{Q~4)=xR}>5uh8cJmKBZQix?slM0F+s}X4);;syeeddbjdSBqXP;}%@y?n0th@V{pS$)NqxUyf`)2&ju8;WXBfNXNzC-ih{Jy#F*"
    "!-"
    "!!ssG|VYoG59{f?y{bp6rwb%(x9`_A3^Z~p9WkGbbNYpnCF6_0JSIZxaA`}&T0e*8uKj=uG9zE3~o`}+1be|L;eUuT}JA0NMaPJL"
    "Yc`>RLYT0Bbq(x=CC);zDpzwKx1KX$!~)AhcFfAy#Dy^em9zTUXEogNQ!_Qq>HZ;q`;7~gSheLFpKeNjD>c<lN|dP1L`T5tM9@z3"
    "=|x<1itdd9cC;eX%0y?B#P-*Da4^7r$``+WLV{k1XMjVE&ZS$e0BxAHz8g-`!ze|qh6+&%Re;{z-"
    "`+;2_2`KNEz{HM)7l_xj1j@pUO+MC{j{;8hV=F7&v*RyzfR(uG(@A!+K{+XVrp5nbaSn-"
    "?muXOZ=PV3gOJ0t%68~SB^kGH;|hh0CDU$OKu=DE-Bb1trn-"
    "o1YEU6WU$_pWC*_SR3;Ly4#4N2h18uWpkbFMp_aqg$rWbNK{%SMiec0|(D!EdB7$@BJH%f7^FE`jI>Ci|)|8`h@ySdZ=+g^icKZx"
    "&{4QoxYnVX?^@yf5mopoG)IuewEIv-hI;@i9da74!zLT^lf*`OStrJF9&q%fxeS(h?lO1vA9g*<$G`QYfs-"
    "=zkhsjWAr57(N|rH2|9Sq7xOiRg1zpiSG=Y!CND@Q^Xh1HU1Xh)9v?~X!Iw`bMej^Uz(2F;qkak}n-"
    "enPn>M{n;=$vD=@;@y^stxCyT17u<Bi|t+WL=u>Q+9*j~z!r(dgsq=psLbx*1VPuR?FC>2rMXTUtNk=~U~X=~Kps)Kk@e|DN6w-"
    "yweS*YzfK<@GQ0_VW($W%RN2_H={u_v#7i`SKies87U%b*DbKKAet5gj}76r(-"
    "nq8r!*F#V9>1T?ig1C$2}7;Th;!D!l5mesQvN_194#*YkY&YIMJ^0?5-"
    "R*9E+OU|pwtKBAxk%gC3Yuf@~X$82lNjD*BTy?^D3r|^>p=<7PQ`nq$I1oE%y>F6u(xAj(e74aPJIXaJL=qz640^gm(+fW+PFB|c"
    "-HyMGap<gt0RO9J%veVQ9)(e~d)qFf1$vsIu?bgK`Tc=Bph_I~3^B@2EAN(=(FlN_d(3??o)n{`K(D%@r(Xr5H)%T3Yar8Bh9;aS"
    "iMnatgeO_HQ{iz~B91GsDE~8@8C3eKgUqiP`A534COO;_*@B8VwDSOo+SA^6_)@xO2(r3MK!*#874|Sz<(-qw-"
    "G#)POdxSt;VE>}ixOUu51@v!cWY=HN@zpU@cs!Y8ov8HPbrN3%2>mmCKs`|fd;R!$6iffR(%;ixZO+<__dUtOM{n_){)$(hq>=Qv"
    "IxAy~evIDg;O!Dd@|ZhE4L#36lD;>7Ik9KyNnFK#(ng#j<p-TD-"
    "6o%y{G40KNIy=mRQF8b!{10rOOal`@+buNNHlsqI;e^|%5wT<??tHAEmbM9x*m`&mH+YO!SXY7M0#cff+z*ZP%_YEi!7xFIQ2rtd"
    "`b(7Z5f){P|(^OI&S~YjHryK%de-*bLPP6mMD)CX%w6Fac^cW?ue2k;fBMP*Hmxys^ZcI(+~3RSL)(f=(2FZjxR<dVd-"
    "%yE$CcVX>@cxwjR8KhE8mY>O}v6ia{q%4=wY|QvB3O*S%B%P(+G9)ah4j)Av@e?^#$WM(P|5{Y$5$IGa?2I_f&xdY(F5dE^xO^>q"
    "0so8$f0NU*xD9B|$1b~0?~3!P7oZjx##8spLH(M}SEG6*llr%q{Auc@6wz>hLYms)R<2&8|jIF;{B#TA!-"
    "sm_S^)?8lb<LKKe6psk$I)GD;o2*SF-}yZBJKoy-3?=^vQ7Tz=Xp}Z|mh@PCGkiAm>4s}C-T%Cnx}%p`Why3;ZzB-"
    "v9O&3k*D6o0Ecf)z^&-e{$`T{%2#QYyfkUGA)gon*b#%7|MMgXf6$bYZwUpv+1XmKs#~nB-(V}9QdZ#Mi^h@*$D(mTk=xC^-"
    "(JN6fI#I#!Z4{u7PLU#eJR*HCUG=R%^bvBS_|#v}Q`B`NXvS%KgvLP%;S!v@(E7kCM2E^sJ&--4%F{PhW@xSfb+mo^_&_?wd_~o{"
    "{)_q$6s<&k-S@HbM(HeaTJ>%z<02(3-"
    "E3aaaUqlGhl4krrqmGVf1Ys!NrEmODETw@QMW0!=<4X|&gJyOo#z{MRTwiWfK8QhDEsu8RI_9_psY~ItZbvhk$L>%mFiI_k?VbCE"
    "YvCQ1ZTBF3OirkDZPJvc~1M$fyn@=K&>dF^QVhDB3tM@^QFG+m`5egM<T;GAW;t=pCAQu-"
    "N(h(dSqWEIu0<2`RN2G7;y{ZaSlooMFZ6w6k+j+T(**`Nbx7*up*95!gevPpY21aTdwz^_rR-"
    "e^^Mfds5RiFD~Io2aY8w#c&Z1j(mx{4bcdz#R0m77L$s564LbI;3;OPPyqvAn+9*v=XV*CGyjq1={ft5N=R>R-"
    "*JasJ<MT^Gx*<AmQ#B7&Lihb4X6a7RK<LP4E30e6FX?=(8=1lVj6SCl5iMUv7*$_NK6-"
    "U}_xjiQ+I7@b)h^e`Cu?Tq{Yed`gRO(AI^VT~&zF9N!h!Bi#N4O9rY!JC5Xx?GHpxMo+o*i;$bT-"
    "wDbe)I2|QoiD>@9-Xet!-;D{pe+4V2fr7C&pYS!h~aZwSZ*P}jJ;rztKP-sy!B3Y8w;<Kxh(T7thq8JAbkndN~a8h^DXw$jIcXtM"
    "(wjrFaco{mzx*NKl>axy^T2c2>fk9_mH#Task48f;w(-"
    "HqF_{!6^|wMR#bqB+jOcAESLoYcipc1Nm3Y=ip5xb5Owbw7KUK`tk%(BPo1rtI)27l=wSBbB`av#Rm4P2=G(L~+r+z)3l?tI{kkk"
    "9(gQ^lEGrqWNiecPz<tbW;e36P$n+{F)W5&Tel%$Hq$_+|4BTj)n2hW&ROclzkj#-"
    "~Y<^R$<(xKfWZkleoUL1{DW)FoMQg=UYV_Q^V%KFM#`Zb?^_Pu#rKKe3VH>o7MOT~DV|6DN@=Phrk^14b-a;ZXbb~H-wl+LOq^-"
    "FY0@|Ee^3}qL-"
    "^&tu7RnuS98Bo!{tH^d)uZ%}ZfKzY&qUJvo7z@~k`a6!6zOeq?r0f`+_k5a;Z7NbWrKGE<M#%vN!Dp>{Sl0tM<)iji!pX#~=c@Ph"
    "QL^fPiRzxtP=Q0yU9Di=DZd0&_6lEmczMm;gB}VVnu*BzdTzuug;Mo{swEsw8g(bfO4nEK-"
    "S6iDD(T56qV`w)Q+E>Jud3iut(OI$-"
    "iDrK1B=ot&cxP<)SyR=qt_G1@=2xOP>G)T2DpP?L2_t1*_q^@Gj&(Q)On2uH+@nGn^dVQBkrI-QP-hr=|rt|y511kN-wDsi~J&$N"
    "I5PXCw<?%7LzhXDS^67PcJ{EDpnn7g-"
    "$gj@zC{Q^d2~%T*!n8Kk|~E<s+Q^(!!<c1tqzOxSZ)SN0(liU$;5~5zq>s{!o*w9W9;i$u_{KnoEJ<MD3{{y?|?UTTb96djBe6fv"
    "`GP?fLHwU^1PA=VXy!h8_k5I1r?&*Jv`S<sPsNAwX|EOLqNtN+rc4ea@F~n{GUi(c$5rURcOSVcR&G)MmePpisum)<2(FIx4bJnF"
    "E0-aU2&YfNj$;N<mcT^F<SH<V5d7@uQNgZfOTd8WfQFa-$NM&Y;o+-"
    "95N#U|moDnaULu!Kj=L_BA{^NK6W)iQ_2c@n01{cH$JT(~hWl;zngYr3Klbo1Fv`KM<5n@6un2=#e2@fnM)ducHA8W+c=h)K}E|)"
    "IrfZ2{eV`=owd=0LgiId0TpKove7{o|4WRr<Df#3zSl?&28tTs-"
    "#$0*^w$yDue0=aAUJ5)$e~Kr3EVL4?iyt8B~qWoKVgJCf73un%Y2X(NOBMsi>vCquz}Qo9~pZ=uMoYk=nUIIt@CT876enl`)#`f4"
    "1y_9d7U$wdwl*>#k;y0JGzTQfl;U`6)n--kiF5`sO&sm*5QWrVQAsuThxdUMoneoDcj&;aJy6hY-N6zY6%|D}C_;Ec8WV;?Jf-"
    "!(opY4nFxf&kFzgI&lcq{p(HZrcg-"
    "e2UU9wBBQUUgGikA(b8j8DL^38FVe5`b0bsek?qf~DesRSnM2{AQ_4P^;^!Gk)y+_onZSe|IZdWw!YY6eAiudOBg|xc)9v4cGIjI"
    "1`x!=+5CDzz)M=n*&M0&Qbu@Gdzvsd@eyU=bA_T=Ijb0*N6#fqcUgz|SKWiXVfVluinQJHs4^e_fV^+~{z{&aM^bC32C?tWKRom3"
    "N<)rKU&}HqF-"
    "$%b!QMXU$&PlHD8PZ2oji4|yltU8wfBZF&Yt=<6<0`|efKi9`sDA5GPUbwHkp)V^S^Z=)ltQZ0GmO?;=fZs=Ba<F&E3hZ9$ptFvF"
    "A}BxQ04vPR~+6_K6feuP*(*I)ySFDz>`(LkqvZmGHU8yDhs%)(hG?U1Q4RLuN12Kf;8e!qy*oc8bAypW==9k`+SmfP_+R=bW#|^v"
    "=+5MqdZqh!bA$FJgRdQsH&AGB{xvv{)Kx+9?B34*fUU$MZcjOrkAf^l!?1@j6hg`=u`mkhV)XE5L5;o<{nx8DeJ2r0p8V>{jPgD@"
    "I=rhH4G%IyM8bl7f8=y(8NW-"
    "Go8d{prV26QM#+8q*41;M#s*;K~J~Q1*+MtanIrnlQ`AYoXi#a#ws%z2wZ?Q|KSDJQ^i8<r`H<@jY={-IRE-Jvd4T}>I3xN5^3O0"
    "=xG6(sk4iwM$wCVs}=^dX;<91Odl1u=*f15L47!`oXXd(ZYeJ`fUaIQkfMr)mJj_!DWQX+f_7EMP4D!8oArTm^h1mV9tB9yg^i-"
    "l30~I>(7Q*`J(%C|@qvP7W(9^ifDd}cs+3P*A{<9O3H@NTt$|1CU#U@`1)(a;Gp(Pc8?Kk6U(E!l;0yY*1;DA_ooSoA!2O~lemv<"
    "Yw-"
    "j)=B3zvj!4+UY*QRl@C%tn3L?9TAU#5hZ$91^gr;qL=HkXczH>mRG^AGU51)<<e9#Ao0#YC%gNR*%%Qhe^)sB(Z66jVvE0iQ7g(l"
    "-IZ;ywk4OU0_sq9=Um;9j<&Ow}gINM%e!bzoV=k4>U#Ulm$9-TG?4v-+Efgo0x*Zeuh`r4T6FWa>przzil5_EGoSlp-"
    "5=T8!JeLkh#f=W6RDILgrKZn}A8mz;E}_EYeqKA&Xgd|{o^gu6D#@BT#-"
    "BwQzzvB+38r|o1>zP&!hYSB~xurpTCX<J`58j3=kzoxsxkh<b8Rg10^pfMBfJc4@p%kfCP^oSq8i=a(tGzj{p{xbzeC9#hwzM@OO"
    "NQ8a8(l@}7<D=_mD8=Y?s5tI*nN=HKx>mhcoq{h09)t!}k@RH10f}|#BI}>(3+mKFl3*OfY+tpoK5ae>&J@>#8ep_92OGN^0V1;Z"
    "l~WbQuR00}si)`UQ^lPM7|<O++bBfIgP+583*;^!0yhr7VdCGi<^ypY9@?E6$>=kydQoovj2J~-42-"
    "3>rB9~IlufmMCV=^2D)Qy&?fmI>$XUF&Qz!}`F@K^XJ)=1TiC7-oP-pgEi?kHnJTg`l5r<X1-"
    "=v5JcIfqhbmR0m8Tsz0J)UP88_1OjQWQ?+R4CLDQTzRUMm1*o+Y>};W=v&fq?4ySq|7<9M<;Aj|I<FTqz*9C>a>si)bkQ|J9I)!Y"
    "QUOT%f3<{kJLdeHsX^%GAPiDUIuA40852V(i%DWQNJV>(4c3`&**$LhYPSn|FSVh4y=}W2!CTx9Wrk@Spq+1`MMx_4!;?xe%4{0w"
    "D?vv1yF^x^aB|tox;t1J9~f|Z||IsfjNM0=(vqQpFwK<lvP@i)AEh0f;j3KsnzdS;6?*(Q~RSEe1EA!#ficRow0%b_?aMCMojcdM"
    "s}f;nvIzvlSGv)BQbwnq3;^Fj}cK!K6TI(gmPR!K}|os8}lHQ9dU{PF_kX$fT&(-lnbEZU`Op0IP>NcGr?0-"
    "{UW3@0@NREw+>9uGYjkcFz6h>%~hLv4zEbJ;SU8wFcdVXK2#Byosl@LUaiUwh9WH?A#$QVo~bH1Bk^$tE!jlJ6T=iW+j>Hbn|UG0"
    "Y*2FoI1hnFQS^&Pp(ad!nINI6vt=7Zajvo);s;e*wrd|pT<3@(Q{;|L8KMjinWQb_NzKIwfDZLYjBLnPFcJXhdGdKqDf|wo4R<+m"
    "k^2B@RDBs0eK~4S6-``FGJ(+(-"
    "7wHigZr>(Ml#HP;3wDcf|5n_qr4~OpaJMbeDuYm!y>uBXc*BoKTpVrlAscWd@_a{6c6f3m5#DOIE#6NKFoq~u%gmra;BOT$ShMj5"
    "VCLbrsvL=K%i3L_yuTD=m*Elv==p9qbDs0RPF?KMG-3rKuU6qLa+V>k0~JV<{p=1piKUX#{w}*PeN6p+L0{gA?q-"
    "&;`l(IaMwGim<RMNVr#a9`Wgp7lS7nK{~iUtzBxtitR{$-Mpe?Ks->rlyZD;NVvB-1f)jJ-"
    "4Z6jwSf3&zfq(G@6x1O}`qqY^;R;LN;Z?1wvp)@~OtN<JPL#HqDKzaC=%u$nL976@s0Ndr(1309F!X<Q*|XZz!)Ai;xzB7hW!DE+"
    "?wkOoEjxMQ=BBX+*~^!%|K7MKVAQ{kXFHqiMTf8J9F06Yn~(SGsP&TerSHs-"
    "Xkg?Knm$)p&|C;&iFm=_^2<JqMQ&k4ipk2Qov*#?vFXbV4@2QpMGvF`{nAd-HTGUj9j_9kLzh!mAb3I@$WuR%%z3Mn(gmM>jEux$"
    "_RPd?Fp1xUgZSVR1{UX9RW*q=>iY#*=X%tjI2`%ygBPO<rmN4$wCAygS3y`*hmo|UZ8%)#lt#Xd9DUtE<Wy;Jzv?^BcE`^!5VB-u"
    "BI-9i1I4yO+)N%P#Sz(>iY;eaceq*vPlrYl)n*4x-L8|E>%=u9C+gZQhC4T_sSg}>8)Y#$BeF15CB+SjZIeK<Y-"
    "=&J_b7cOF^JIv*u!G_kT*&lXg5#@zpE#P21xPIs~p{cU<Nd*KKj3rER#qaia{Sh95+9MAcRfzU5Z-z$x$jmHq>nz9Mv;dG&Ue2+-"
    "+(gI<<_Ax^x3PXmaH8;<?J66gmp4MTyfbWUi!l*m_``hmN)DR5qX10Ap0mFES?7f}7-)57q%el=cW-"
    "THbp7rw8Umarq^ShYELq(Hi_)F<R+UNotT8p(h!sFy6GO*i?O}fR3{<cmv>vfTrzaCImP%32o#6Nq>(fNOzX@{JQqmchR-fn`P#h"
    "MFt43s@2ZDQCS=)D*p}4aWgPwzE^{$W=O2&H}1&*5gipF$|amJ;2JXQyu{QWKTi);#en{BKtv9hPiOpeSWWOR;)63(0r0vEoo`L&"
    "E^|fZ5IRALfI-crkGMP%9i1CPw+95VNpN~&HHy$}rCp}RDk8EADP5^W)0Fcc-EKxis=JKJ1z=<>YcLJolmT%`6dXvwhoMU!e=-"
    "zsp+W?~F3Vr;CaGgZ*N_LWfw>K);Sg$zj0N|O5lL1bR2KuXq#6rlvZ%au9zmiN5SWuin1w;c?12GzivSO`2wqu(V)_D-"
    "Qo}$^p@T=u95)-FvKjWBbPX*-^kG21J@8(Dht-xZ1_{uMX)+@Z>BdpX!d0a3%fq7Bt0>?(;Yor})L<%j2gCnaa2)77Y(7}Uc^Q?&"
    "4r8VU*876Aq$lMeMuYf6=*scqV;ryk25R)`ze$O&7$|*E;=r+*$C;}56vLuUfwx5??4WUiDaYedqei~|2hu5AMKBot?MERSZHRJ9"
    "qq0>qrk)_CF91hoTyQP3(9p?Yy0*ZZ%*_h8$Xp}pr1huB0vF;OnlMnu1uE)RNL1UtTcq0qo1ft<*Qyuw4k`m^hRTT%`2ZrJBAp%y"
    "yx5=)=+jUkF$AW&Xn{>C?o*i3JHY*G$?NsB?!hYszlx-"
    "%HUPc~tqss$U_+cV3SbJFARSUz(F2hoXxTF=nv6*?$Amjjb4q<(&dDHsCCixE4rM>IlB7}4<dDqu*pxQnU_5HVISr^VB*<=#E4)3"
    "vXpyd9{KIxTo+Bjuq-yMfg^<;64hp*2GCHGVN|H+Cp#%8OKwF3X#Q;se7(Cg)=y^qD^C`L1Mm6DYC>1<EeuWWi=SRl9e9DN2lce3"
    ";TM_ZAlHb&`GxP`mp;=}wfE{{xu2jz^a*~N2pLE6jrsO65C{RP~%ido<s>2w7X+2^*uRo#@U^3NWBurwVTGkJH@P^|nk7sxl#?CH"
    "92ppveS;4&asF(+kFKl>o>~l=T`z6i{Gqqx*%Z-"
    "@9zvAY`MCx+?C!b{|=^b1W3YlB~FXsq~ExM?Ojs%#qCr@FRbPuYnfS8CE$QIMsZs4-QlwKv}`(CEBN2#-"
    "^Qh;e7wvGImTot{s1Bo)nr<BSUSnZe)l5>p)T6?${r!v*04O0EeSQI$I?~3C@jHMyZYY8uxi#kc1N*jq_YzoWCnddj<`Cg_tT>Vo"
    "90Pz|Q!zt(hoR|Iwm{1NeEpE}L{8Ti+(KOMmFj5#k%mBGWjiGtg(Hu`#3DatWJVIc=(@&MZe{+}s^l$ZmCj-"
    "8+4PKm|`&YwbB~Gq-r_HAzNG=EmX)&i`f)~Y-"
    "E5ZTs!sGCqt`CL`N)N+TMPfP%N;D>VmtPjBhYtS$f304uHsS#VIup>HPm}2(6(Pel{nc+}{Lz15a6$F^0t<s5g52erbyiR=e6j6e"
    "3$;*?rX~m;1A8<bJ!jsZM<Vb!z~b@)(N0{m<j=_?tZ1{5#GtSk)J*{wZfAFWh5>`&CXwGShN4^rdUgLy7EpZgfqKv*k$}*795g6x"
    "WUhiD2kB5(CHgqV_T5wt25D5U8Fv+cn*R?3fuI0T9q(nNd%P&$7Vn-"
    "|sdBUxE39)%cob9wJq{~`wQG(vBBg{Mbs#sO@uG2GQK4G8wgNj+)J`>$dtY7`^X=^T8uTF7tPD$4RV2BDv*$7x$oeKAz|49=SfRi"
    ">!D&?`s_&#OSj`YKFc<uA#|$U%%u@_SeX7oRBLm~~?-"
    "*}P@FdhEzmG)wOrl~^b!r6J%tfR{N1JwIRKws4PR<8s?Tc=eev6Y1cefb2&?OYbJYV9$<QXDC4Dgo66m?|hG~g%dJHcWsj;*TgkK"
    "RG$;)A1EN(&AT2kV%w9Jd*s!iK<2J)UEx1zCDAzSq_LqThuAw&<3+H7*GzQ15<lT`3kL7lBICIPnwmp>>`=k)!t?4nfv3+t`A>U@"
    "#-)f&9$eA{g3@bN|eMc~mC}Do!)r7s9n7BOE{j<!M6+_4QidX1Y!K-QUPbi-EYsq!4mR;A0my-"
    "3G!P$%RvpVru0uaCmr0z*=yXK8oiXDV>&e5Iy0bB^U%#gJTMb2_;g}AO<5#P<4|Vv+9KOh!nz&KmuGG!(n|;5GmpfqQK;Tp-"
    "}z}A{c|{aluJ}*POkk^85m;P*XK>l4cx;NV92mQ*$w>%0J8QfU@E*P9v-vAUem=apa-pU$`?F%}vv6{_3T)?3NT-"
    "5k|R`Fc_a9%5^HAY~zR@utXU>t!yfYH*k_5{prAGQ^6;uJ9=m+L-4WT<QjyXpy%<vVQYiDE)G7!&-"
    "g<qe=w8aFcqm6jAKB;QI5<J6&LoQNVu?+qpW*Y_vi!x=H^jDslZqm+U6}1Ya^^R$Owlgm*O;@FG!Alh>lGt1QaMZ8!8DpbGWFrH*"
    "g!YJerJZD0ayGz)tnUQ$V7Jdx#$(i@+;63P6lT^W2vySq0U=tsEw*sngRR_k>X(Uo-0@E#*KA>B=KFLUAj;{-DODcb-"
    "&RBUAf<2LUQhkcrk{dHVC<@xy)Y6<s`hS8N7pnh|ReCJ&EOUt*fParhr#WCMC*LZS#Bbc6O`LXxAR#%-"
    "g5<%A`Nk<kR2CMJYCE7Q(XfCORGKK`1zIL{KGMJEo&bVz@SA_bjvj#DRN9nuE_)L^z=7)Sh7_(n;wJS1-"
    "57DjDFsi!<a2uiWMP9;4`Rg1m!kH6&gPVm;{Oux8maby@KK`Nw0)AfkHje3CJql?aki_(<t1tv6bNz845&kvM8=&YqPV-Cw9Cp7k"
    "66#&Bu(-"
    "h&1zxv$+tZ>!miK`1?20a{7;>?$N@$lL?1VC}AZ=h!42_ffbm9(BoNuNoP8NDf@Xp_yY*u|?D(AP*tsxNw}so~@tGcnFjm_hM&&8"
    "j#F_7T5TrkGbU3B|!u8Sq2t)EERN2^>BaVCq4k&d?;H8~7CW-XL^lK?`mg6H7qWth!Ej7=BD6U?miHAA}0(7f@%i-"
    "e5os_$+&w@)79~4XMNJ0mo0_8!uua=3LosF<3^|N0+p2G9PJq8YOEp7#$_nU|y}RlF?9n8<UEZVy>BUXC^IulY@Cg$^s{PcYWiHD"
    "x9hW(SFi8O#3u17D&URG@qRfy~Q7_iXlZruq^$``;}~s>fTL|u17RSy3w?vI<D$s6M!TjA4a37dRvlA2J<P!0ll4@3czzMu7cOj<"
    "g-"
    "pRnwjrffmQV|wEK$|n6Q}IphgmXmzwlZWY!ECRjvnli`uX_!qBW~F#*$Nsm?fg?yqULVi>o9{~+{J4)(&ymzi7oUHdotgh))M4-"
    "I^cB88)SV$gQmE%65I9bF5^A_wK{O0)&og3kJ_g2mnZbW?`sNQ{c&^HPvB>sYe@F$oM}sUhkVqj0eiAND`GI9vs2LlS-"
    "W+va`0X8aqCP^#j<%{%(zTjG3<aASh~mUB`vIGFMrz|IX09A}cr4Pq(yG}-"
    "lDl+DWF)Q>ALyyj2qsq#m9N?w|(5FKZbx<^jPCZ0q}Km>p2;SvFqd0jMhqy()>S12pgxD*vxHa$~Yp!PiLi~p8m43g~wL?8Kw!#r"
    "S0J0=?y=f9W>(sY;)Q0WPtIE6BNjg~xF)nm^v-"
    "lQiF^rOMog*h_~<`iRcqkmn#`h~MHQ8Lv2nW^|I1L#C30(3{dB!K&%1A=EM>Q8SiE_GtFWIKTI1448OhBF5gP-"
    "{>qA?Lo(nPxHgzzPn{2UEkOG7#pOA@Qo>6ka3%W?FwQ5hn(M1}2q*A(Ng&eh4vS4)^?Tl;#_jE5cH$N$~0-"
    "`T`={P0>$ro<kBESd~@RI+9uFyq_f<0HoKO(U68z(Fnp!i{`BHKR@GGx^s!a9A@c_3-"
    "e&tM(qfJMORqI_9ndvEQy#ze+550doIXVOtiO~wi&I4>jR4;P=i;UMU~l^`lvm)Rn{FBn2`2}v$Hg-x?Z4JA;-y{CM-"
    "J`@1Tl{WM@`A4IU`zEshgoDU-(OBGxMcKiL0({c(`ozZ8HqvWOBG79IrwamX28UU1{oW|37WaS4Pk#Y_yPCO3y-"
    "vj+02iS#3NbaXaaT9+^YRHD-jfz!}4Qqsk_uUFV8($@i!H2o2fH+kp?@W7qOw_h*+449)phel2!-"
    "Cg+V^jZrQ<;^>XT?4ZbUdXKq7ojhdamb$cB&%mI>2?5XkOazBz{3nxSFkR|k2Bj%8bgxxst6pE5<xhwI|~*Lqhq31`SQwlxB>@Cp"
    "ZTXrRAN5Sc((MiGlxfXA;xtPI=pQ3z%-B1GaB}s=9~BhcSg4;aX-jE2jd><rvT&!{xd`tv+i#)I1Ua6Oong9G)ix#GM|kX(iP?*f"
    "MC$hkZFyCdvpb?%7{s0Ts6!4E(0{m={RdBP$%Rj{9@uC!`Sy*Me*=TX~le<IWG(s#T6Hy6n^$KQN>kgg7Oao1e*QNrY!#4(-"
    "Ko%CZ(=YwbREu>UvabJ)jMUF$Zo79Eu2~#QJTm9(=KUh)~=c99)pe`(s=$=VB(aU#f^uU_eqvaSKt#H8V*3wTfyocL>-"
    "4V&Wn~0#)W4c<@unH9&Y#!$ThD0bDwKD2CAjPq2Oc>Si)dAnh@0wUtc!%2x>fB9BOyCzAOlbzx^@m2kvwXtI`NcdMa`(gl(<Q5mk"
    "CJs1LWP$ontXmeNa<B@|?EHQe_yJ4t{+2b@!lsWcWLCUo>VJ6IIdKnft{MbNrxPyk4(Naf$L-C*#W4+r-"
    "xC|VSDFTAT{xezt1fcPN`@K$4gb{}UqS!rXrJ7@!rT{}a9SD?&bUertDldABs_-dj41G|dlqmZ{KfO?GQ20|OWLboP7AJCBpA$Wt"
    "r4dA$&eSBKUGN#Db+jm){Ky*27ibbNgow+3xYnDY8}iD;W-"
    "WSBCWNem96F#>CeC!%v`&AfK_rmp#Kc`(NilDXkvikZQ$LiuLjQ}dW)_K>+yO?#uhIL9pbaR&z#Q^Bmo0c8@b@MOpPXI8J6sGvX_"
    "zdH$RV3|ZLk8@flMq=4<_O?klE0sF@3zlEz++hfG~$%16C6tdOdYGH)q)4>l9G|)UjdB!?#I(QOJ*vrGW5oGXF1H;$|^!mh~W0ym"
    "f!9RIu@}+p!}K$XNmJ&@FklG&L<I`>43ESWpZ+NxGZB4Is6^!bufi(1QTr`T(C4CXdfDYx+bzk*o(mKH%KbQ6?by+2w$C-"
    "t*t!uRtY)#m8HX@B=G-"
    "k?|?RSLuS0Ng43;kS!({jDhX%pl5qy;0@yAIQYyX3A9!Rgt;am2%65zslIXqZ5)if9Xyp~*SC!t7LYc;0v~%G%pOVp*Z*!BrY(<("
    "6}l`r#+_tZ2a-"
    "|AZ&2;Ge$xyQMyEEMbd7W<IZDQR!Eho_|N5Qq&uaKFne^;LXjGHf=gC8ZmMl=d7mXwQpjEbDs6=>S<h{w<EO;(v=@XbH7I}Pcz03"
    "_`X`pC3-;V>Qx({-"
    "XDuz^*7*V*`Bk9YwL}<+Tg0K3aTEJ>2#>d8kDIk<<Z}LSZ)=BWbNFp^}dcl!`V+?z!W*Uo8p*hQCCWl3s&X6i1vjSb@HMAr8;HYR"
    "p)T3U&rvV~+2f6(@r+Kb0D28v=+y_jF4C_I8){?+V!*A&@+Ax;d@Nl4b9M}cjRA}ROv@Do^W!^-gVXvjb&|zms%3x;ma)D-"
    "(*x^P4dqE};H(xmMQDyU?XiQNcbI55H>cb=<0_p+zMehb=lr;!3>OyBgtg=>yFOV-"
    "E8D9G+EQUcqgF&w!^J<pzc{+wOUj=DyCxo{V+=J}%n0A64B1%c?F-"
    "G=q1;Jy<tnDYff%$`~{DusOUYa1n<OCr{<O1e`>U#AAC^AZ5K}iIe7vSdQJ$X3gFx<DX#z<ckkVsux&UUo248tBF9xjd|SZceu5@"
    "=UjyCFG9#L3Q6)^PFvq8p{hL#RlRQf#u1@65bN9~^KYTACoHF>a#YOfb_NRjb(m(C!aZ@i+!k07Dk}QwQ&8ns?XITPPUa553s9<)"
    "Q*61K;gKr323_e<%h0riB9>UNszM9{E|(QS>PCAoENeDg0@Eu1q*U-"
    "I$!kBN<ldfJlZ~EE{xeB|*pfuTHg!9O&^ue0%Q?h6h;5T?bu>pNTZktnb~@Gl&5!6n?w%%{bgN<@&v=T&{A+iI{BTztKJmRHG&F%"
    "oH01%7%2y8)iKDW>xE>UgnWeR1V@NvVNm4$YLfz13E{IZS0aB89a295*1<@Q#Tk=34vg7KxfFzB$^G|?mY*Pjs_w1wic@_UU|P+("
    "li|c`X1XgmVJiu5lmf7EU7b#n?pA1i07a#%!>Do8W9GV!qQg*8}#Dc=1*DJLIuZJ#HM*+0mk^Klh>Z>lrQJi9Woi<0hn9J&&aO_X"
    "zsvKO>t|3OY#(Km?4ie?}XKe_Km($zc7uxXbdcwvuD_@QJiZ~KwGFW<y+Wn>%-"
    "T6U%8)MfMGD~YrB*{;%o%h?QOOqN7EU=X2&oE2c1YtXhD`IgP=^#efZLK6U_+$KTMu*1;l&NU!A1r(4TPsb@@z{$nKv!%uE31*5J"
    "q!-Lc%k3H2ufqY&E`24fGjO0Et^;TS+zl^JwzQS8z(m05-"
    "U=WazY88EpgX5riQiEtXFPO}p(4w5k@O{E30e=0vi8fhdl;Nm=JmPIHDc%CBs+S4K%#|wN6Z{0VgIH#DOQcBRdhaDR5{6qOeqDh7"
    "eePj$4K;}x;pkIMr4-pG~c2&i|B))dcp}5ZB8dVf+73fI%R3MB(tFTNCCh8@-0&bp3xhQuEUr`|9FJErJ`O4PifDt<zw-gmOXqph"
    "E2BVahL<)we!RSAur#6l80`gzuy!0<gg5(@i?+spd=uzQcAupvoyp@#|d|k#4`z|C&@Y&uGTEN;hFQk0NCb%=5>UsHbFry2z@x%E"
    "E_+iBJt=cQM3Zg$aU$Zo6Cffw^0@EWRH_k!Py&wR09wijl=o+<;;{0MF(K7qn=K^rT)X7oSRqo3$HypI@!5(EY4_tp}Wm`6FZpmr"
    "ydoj{5K+-dWK)u^5;w!YN<w6RFiI-9a2&C6j#_wm4g<BT@A)>yPym;&<A^kyW3*;p&hZq51dtBbL7*wDZ1%oMgha?Q-"
    "zHixXAjGDQ{=O3?X%O&TK80{Kx*jI=9^;S;XTb=bXp;Fz&a23k(EGUt7=*5qgAhAe>%7=f>J;ooUxw_Ok~l%wYpwSfh8(l7Z-"
    "%Zo`1Uqw*(F1s!$T3dV!bUmykNK(N|%4;z(H>FgF$5CbuBv|aj7p2egYQM-WPuzgylIQfr>Bzc>clB`BH{66@T!jZX(~%=`7V?Ae"
    "DWkd3Fs{7R7fOFT?A>J_7WhS1u)Mc)a+2u2X==cmn|m(}T?9h%t|497dL@_}y`KG@1wuajP90fcy`R&zkv&CPXrD4dtYhR9ss*VL"
    "$SLn8vk*5wkrZq&f>bPCTpz7CJa=%xs-tmN4<Qayy4W(~~6*tIk82o?$4Y{u0+He!xoj#x~^(ub^H6S^MZVh8qVs`4n^vD9Y6(il"
    "()yI~6z@_*P^wJU}ANc<RYZ$D{@I4CIb$;fdQjqmwj&=u&4^3ruVo+1cMF2S60u6d)8V1v0^abT0)f|5Ocettw}}iL&{}5yLZyp)"
    "Ceav-IXXZcNN^;j4zXxw>11g%9lIfNz#*N6XGr>83#eWVNpXX3jv4-^=<;v?+T;tV?Y-"
    "M3VSZ&nZqSAao2=Z)m4DPH5XN_FMN$EjX=N^B}M%57|D)3#U<_#dPp_xRo8dHLg`IEe;{@Nw79b*LIlZ6jau6GWd?X2eoL>(34cY"
    "YIUHhaL`GfWt5D^&f?qfrpwcWi4dW~_HJ3F*&GUMAaf1!?sD>%tqOEFMgnO}e(~VS$>AtF46p#Gq0a>HVxMukaNe`4S-H_H-"
    "%CDGONk;RJiC`pQf?AK=(*7IJbq(>r-)|A26pca9=rn^vw8kkegC9WPBb>4-RM0MS;iI>k3q_ggi=69Y$ZH-"
    "9jU0oGZfhaUOWtLqW*7w=w<@nvXFzp(Xja4rYQFmem4x6sM4^kSto}YNR^<EscB2j6)Z<)pfF?luM?ziSzk*?v;BympV{BQX=Uq;"
    "C=mVvb48r=W9H68SvQXPt!}_bPTRA}442N+iUr{F>YfvCr80sCzwcJDH{1QY0JHj8r;}Dbd2$ruo=t5}eX|j@%PgAL0sXY6tMlPD"
    "Le@I1P+lyA`9;Au&+)HOtc&>ywrX*QP@{lhQONHXEhlIrWN+MwUin=5I-@WI4Sz{%AlP!X?24J-"
    "D@XzTFSyj#N^S>6eBH6qzQorCOhxEP6B~jzzQMJRHAq!Cj_3eD33f^dnIN_xj7IQ6JOu_$zA@&|Na4&?NFFq06!w3%P);uYLzZSn"
    "h{Yy<3zH5r6E^NK$#6rzvrz#cQ>8zTI@S&!UVU)x%}x%z8po7@?#Wr|gpRDb2zNqMz9lq#V+C6b7mn<5=qxefLGIOZ7-"
    "HDM^reHnM&v4Ux&WiRO`(|k+jDGCD;k?G4<Qt4h_Bt|Dt{I!jM&S7x+BN<(=3u}tAfgEILcrI33a*5x^OXM{<N(mO-"
    "VZ36wk7~H6jBX+3|+c3+>)t(_k8Y5v?g>^1749irQg&`eb^tOM>DU`vJVs-AN+Ioi2KS#?k}>aRFvgLHbQ6Lp%$S(&&|zU-"
    "z?>j%n^-f8}Lp(@ijdkVdBJ7~RJ>Hk8U)*YQpY>|?Kmqk)>hy2F<X?a(XlS$CTHfY+4qAp02eGiMtsh?t5SbHp7UI5o-"
    "(ROi!nql|rDIU->aN1$MeHBOQ&PRt3gfY~#uC4A0F-DKSx2*qcC05g#XFX6f}B0_9GO?-9|x1im^HSjQaoxdv6+WgP{BoMrE=-"
    "V;>CL3SmM`;pms{6LCr3@{Pja#8%Of$rik-4B#T+9WsDGwGkN|(klcxe{|;|n)s<bC-bD7YEK;;K-"
    "aEBqk;(bu48Fi0b&jeOz2MJWN_7R^N;NmK%y#^}u!ZVDBp8u-"
    "l4LbS}*nA(wn`!|e5jOD5c1}L`W3us<kiiW#9a`I(kkdRY>P(f1O(12TdW$<L1JF{8FSk(C6v%U6dH5}l#EIZLY8>1)Y=;6h6_~K"
    "qgIg6zOsy&a|YLJP9f^va}nf>qmY^X;LAp(Q}2f-"
    "X6fAd)kLLMqVR8h~rEeZKoSHx27$p|fLRBP7|ELcSn^&k)Yg|6b%u;E!|j;j?3Aoqf>LF8G;2_diMP_zzAI8D9nxhmyg=Jat{@<!"
    "NW811kfatr$9DFz2a*C|<Gq!hg$>?m3km;p|pdPIf4dvTulTh7-"
    "I)3PrB!Z0^c)doH`F|%scUNHN^OFds>cT&eZ0VUi{YWn71syzS>Pec}y3d%w#U?H2Lk-"
    "+bKLEydVRoZON1Kxei_AR#|#Mc`IFF@B$S|x1`CLVZXr0|3SfK~$$p7+<Jp~3>3AZ>m>T!yVigS85@Xg}+_iFyq_K6pYgh#_m>y$"
    "$;$9r3YhGpwp?OqBY0zGKHc-BKm|y4Yo8T`X63Brj~+e-"
    "oG2T^R*d@9=&M1$826AojRl<H(N=JG(hY`4)(Ol@Uxn9*xhSOf5YM^%X(lvh-y2Y-tvd8NC#=?c(n7hmg*ni-"
    "}eq`iw7zVTA|CeG(#vkJC7`=<8SYezw~BsC(xRQiHw)GJNgNkYpF^mdr$LzleYr9;h}Pz_(snjlB`#a>(%IbpzmN><y2=8e$B(GN"
    "fSW)AlPO6T1C=tb<`rmoAx*-ylM*udGc0@jf$d<_!}YhLn&3!<9S1bSUqPIgGmG?T1ESMeD>{HGhRwM5Y|sTs(r-"
    "sxQJ|vTOqnEH4YYFg&T1Nd4NK4abAXXiPiNlr#TEOEOc7#?lb_71$c_ohdl=h}c{)?bIFM@wE#E_Q^dETsjK)H0OHJQ@*c7#SBTG"
    "yZ^J?B5iP)X<h$pl&}$v-5PqbXb^PSXae)`mVyaP_idguxBcZ3tHlA=FLyF24orc%pGp>L)ycXZN(ZWBP~k6mYu4!F47Lq-"
    "jr;Q&0egNPu<l#TN`MnWbAv!uwgqU+SkbnJlTo&L>JTZDgZGs72#kGz_*Ff=H0Fk;537MQ-BW$AUB1}c9_n)f^ibIlFme;;I7P#^"
    "3kaMFP^%b$a#HDvyugOtBgrzS2}EuVK3j*~ZaeU5yG>xr1nQYv8qGz{0+11`ZreuIah3bOWu**QY^RnbDctDC-"
    "?Hi@x5SYRAv$<@Nd`%Vr$~dEcRA%;2ES`SjrLo>L0?2h0}#5o@<@*f=bs5GfLtfVzDff~BA~0GWeM&J@`>jiX(Q7ZO(R25n0zm7N"
    "VoG(_X4J#{s?GuCWRYI5)x!KQyVDsvO_Ar3sYNeqh(SN+SwNfI`6KroT}&wXV`J{S4OMs+JZ62ROf{#mXnKNBAYvAqCmF3hqGb$d"
    "6)a2j}(Jx4qsU9O%uF{`TK>@cDLwq+aINsgpUJnu@atq2{?YX5r+B!?wC5_Lr+a5cTvO?blhzr=S3Rk<p+XFOTTaodPRoBh9B(u+"
    "UdO9=k3nbksyzs{oZzTKbGq=ZO9I(8-hZ<!iu(<nxYECJ)^_WjIj}sh^miiJ@Ft<E@ceh-NF;CR9cXy>4cHW)ac@ayNW*%@~d>(R"
    "T#l_Na^#B$l-eALDfikR3J0O9S+~j;fe|lnU56Dwfz}IB3d8rj{-"
    "<EyNg!=Dg*N&$ra7Lr#G5);D>V6MH){q8Frrue+^rZz#jw4F9quK!ZqCeF|aS`yLE}gQ6Q@TLcHi<dD{xh-ull14ZZ90BkJkBVv{"
    "{0A*rtj_f}UPkCA3EK4SWCV_5&=GNUBxCZZ(?Ud0a#NE@C3p_T~&oi^SNaIe11_jHAJTxs@aE*{Z)X3{_{FRge7I>TBCnei>)Z-"
    "fywdcwBKL=8JmxSiNX?Dn&q4xhgk%UfDn3MtTVd-X9}?9%z{FXYuySX`Ve>NdUTu;uKRZc$#-"
    "(7d_l+0h}%5y<zv`vIzC99TQaxjctLwlkP4(^nao5T5gDT>3LYBbmDYv}W|L&nDOb$l>;jhTCHIaI5sbKX5K<td(Ok%YknXR!Mo2"
    "Sur1dwUBDC6XL7-"
    ";#@nvA1f<%1{Hoj1~bbv)S6;gkOM~f;|@@jJ#V>y<1n+uq#qyKD*;Wd{_~3La^k_mNYh7qS1>dFtyMw$+X4lKep>51aNWS0h(@`W"
    "cOI7fjHEn$hLW1AtN;u;Jvhg*sHk)ZwvuK=*l!ckVd)*LJ*+hUnr<)8vJ%RgG$-"
    "T$^NOgYa_E65hr)qcYDEpE&&4x?56ruan#I;FG=TK8B^YLj1Cma{$gU`fV4twtQLPpu2x<r|Ar7D`BxA-"
    "%GL!K2*AGL+a9x!a2L6db8`>S-8PNd&GpZ2~%<Sea-rAmqR~^{+wRj_N8=EY$8VpyOXx4^lboLHtX`9aAWkJo|mdz$g*+a$qdC^t"
    "TmM*eLM(J2Wg;T__euphSFlN%GpI=C@x$xm@LxA@(Lg+QiFb0d)OW?X&rB*erZ7@k%uG`C|xjCP+85Mn1R04$e;YC&YMoB}Y;D83"
    "r1MBGwYeEb&CFn>y))H~lS(CTC<*g(O%itNfEt`1O&pmO6D>3FYJ@-IP>~*|>f@xl0Bz7i-"
    "QapZMg~mE}|D#e0z!WqPq9xGWtZhFILZ5NR^v-E!JuoH*22HI-7$Q2Bwg_)*?(w!bYQm}z?8jyIEp{jE8WS-"
    "c@RmYM592=IGX)l}NWibVS;gS_Vwc(9iC+F|&A=@AINP)4RU^DJulT9jjT)RyXrt)oJIs$gvH*6?6){i}y?7Rzfj}vMX(@F*fz1A"
    "FQjSd%dS<StrFkbd<Ci^9X=j1ZaJB$uHc}GT0XZJK$TpjKeoJQ;3Kdos+HRl-0?$VNXgj{-"
    "ML{Tr!fpOV*)L7zP0asWN#X!JU^ke|5BsK;cd>$yx+k7Rnj7I}s@n>PI-"
    "@eCOFC4Gl$~iJyu(8W0t2(&lc)=<x^0{<nGKL7u;qmhZ?mrOEwb{G))a=*%&n+y<?ptMMEhwP_{Fd)&~zmq9gzh}5qJ4>WO>Zxpn"
    "zqOLpgTvB5+r=V`fRuYzlC^rx{gCE*C&&1acSH@K;&t*;~$wzO$s{kTRNkrl`DfVz)ElAN72RmB5vdsI9E4pz~tZWw~Xx1<NLC41"
    "r@|cF;~e-"
    "=vQ5w#m3r$U8pI<|(qH*ua#Qc4+#2mhBK)KcxS_V=r?iypwQ?hX>wk^Ch)4(^RNyCC_ahoPs>FO({tv#A(27yjyCoPiPX6;3)RDm"
    "Ls^w8d<#YOe+m5H8*t=+O0zS%7tLquj#Ef!d1CJGtZcwK|s3u=1<kkQmD;yNiycP-kUT=8I-"
    "pNS}1fqwtv@d0=U4F<2sF3pE6ZMKSLEwet)%*EA1kf<K0qJW9-"
    "IKJv<_Mt;#}J@c8afG%{y?HZ4W#Ps3ib?PpuFVHtfu{`R^@s!U4<`k6kZ(Po^QLdOxti;}gy44qu)AOP_8$>9|)+&-"
    "pyP`obtXwu@}3+h>`!#tS@MoMe=Z!mjoRs1EIc)Y5K=7%#E^#>#_99oet)f|`sHu`-g@~tXvupc0FBK@cZw-fmnYinrTQ#cScC<|"
    "-Y^mmIu$c<yCSocoB00J>?0k!6{mct*0=N_@(YjEnsAR2+14*!uJxqMY{?DOENY#~1D@{SkeqH(M8a)SDQw)lJ*Oq9gfvDqw<OB9"
    "mDi>TvaLzVeH_L}V9!lC^X$C%NX;nrbrH1K)*KU~%m0AmLZVQ3Q8j(e(fP;uYRa;E9`mpO&$u~}F*CBPOEgk8w#p^RYS=r)Px>~&"
    "u$h6Cb78wTY&EFfZ+E4Hgh5#sOJB#VA`ff2Q2?uHT^<s1X%Qwv(@7HMI2-"
    "bR}c<9lW^*Plm>7wA&fK>^8e7dup%{V5UV*UP~H5nqpPk#P-"
    "khz<0=&HkL(vgve~$%9Ow&=y*Cg!OjH8nc))77HT##6u#N0%N(eEiF93>cICG0w*lbQ7rZDv}};q6x`&eZPRvu%wH{p8?3@cN5EJ"
    "<D_9Dh3;PF13F>{#_;)NjQzG26jIKBV%i<H89Oq%>GrNMqt|-ze-leuy#wc=~v={cePz;@*5WY|<VNi3uq0b+p<OCxWL)$xeYt_}"
    "q;Bk4Cm)&2py0E<iCUR}oqyM}{%`7haMAQ;^pnddiA%Q7vTWp?37zq2Rl&2U8FuU)zY|Vp5XNbori<in8)!w@}2GAGP1PJ_)S*=="
    "$nQ9E=6?6rNP@s<|D$O~8GSS!*QNI3cbyaBtT(Exvbx<o%Gwy@slKH1b!<l7-?@)jcW?kbbN-"
    "r_WhSePQ>9w-qWkfklYj4rqO63uo>$+W{;mgEVqQt}5jW7aumMbg?_-H9_ujkouP~gJQ@U&dT!+o1pDcg-"
    "EfZh3G1*XU2!Jjc3n-vj|nR?518A#8os-PpzN`SE&$mZd>*Yvo)VJQH>dM{%-G>e!WkU@i8!XhUsLu_*Tda23r;%9{J-"
    "td%b8F@XZ#qg8PU(K`)q&gg6vv2dxW~Vv#8=lWK0|NBI*iq!JAna!_UYJzkSq^*m#4t^n6hEp>LW5`V5(_dSaI(n~IM9nsVwj5z7"
    "|ns#A>`L)*$<Y@kf&{<5%UvtgP8U)V@RBs1+q-;eL<5XZ1P+40z7hZHb2k)2G|(&-"
    "v)whxBcmkzL(0DKSfRHsRD~OlX=J5JHU8jSQ+0Lc90P<$B!3r40tbFh|Qm&EGsGKq3t*NjZB`<d+c|JmOC%xvn;(cY>$=Vx%q=Ns"
    "c^M|r<^wPu%P!1Q@1!8cn&Q)x0abbw!v%zCc-"
    "5r5~hP~ttM7l<&=cvM>qG`SfjlB5|w9gy^80)&A+{NpF|#*pe5W;^iiq7`@Bew4cf!IDS}@xT&k?&9f7rAl|klofGk-kvS%cQ=*D"
    "xq9(c|hmK9mVQTrTW7!K+w2sy8Ifw_H{HbY}Cd=fJ8S!rH>`k-"
    "l){V3#kJOQwuNg|x0D(VKeYK#YBGPiATV{r5b%h2!AK)dy0iE;TV;4gtmK1*;ihbS%a$WJ_xC|R(4qVQ$9I?5N(O8IGFSW!3-"
    "=EVZTO$hv1q;2b$Up&%4wHphgNDRll@XP%WHVxo)pzKe7OrRn*yv*``m^K$@2Q@6<R?9zNNI>nURNMAEvC0x=qOI^S`?<A5gA{vB"
    "C+aN`+zTOhyYD@<R8HY<g0B;AN9{#A_R>05Zn0Opcw$udcH3h=YGrvgYudoz{MM8(0ymU8?EqCPGVv~j+u`k-(Py#c2^SY?qH4a-"
    "ys1CMdEJDYS;*T#9<?mik4MJ>Lj1_sEYI8{@MmwQurDEHN&aI9BG^{rL2|Q?2K`04PlLUR!RUdDUofikK-"
    "u^dSU_}4b;`ELyDmWITA_>741@KYP^WwR62sP)3k@-`>L}>ZU=-"
    "*mzw_JCJWSuGRqDO)IE?TaF<{)WPnlPm?m~_lxg{JW89D>Id49!@^SWMq&b)r5!k`&@`jKO$BE*ZV0M3~<tWf(A@7NCp$m-"
    "(phq({?WPORGBe1lCt~=>qXLE2dzgzAr(p(1Ot5=>gVJ%*VD@mIumkBxpANxmcGv=fJ^}Z1tx!IW4Zt{7rJ%;#k4EMsc&s&F((2?"
    "WHl4*D=ZTAkq?@Q-<8}gzJ-"
    "}u3lI^JT9Fey!y72H*?YqN%W1a+R+L;Hx|*H`wYxL6}~#=UXmp0{QLLHn4sASlA<0U*PTz{QCFEq3+Tks@)b3D&c01hX1k5J#~wy"
    "wZ!&DolwwCD#klmAaU<rBdRO$UNehU;bh9ZeDaWl)<%_BJRX7b?TJC6OhKWZ5_r9=^$hCbr!WB*7eSOxiERQ5pS|et!<nDKP>b>B"
    "%Smey}`wAxfY_8Yd-UO>&I;e7mXK_L}v*2Q8^O~J2|_L*<dqEBp@edMT45RoU8XVndL&-ilq@Bufy9v-m%E{R|Z-"
    "RVjAAAmgss0=H)D-8ii}C6|X~O;#V4%?VXC)R#p^2h6o<SaF01fxvX|?B1M;*@GI_@Y&4oSBu>vZUrU-"
    "K+pNyq&qp<3&$?&hoMJTEtcxd%QCyTX)&Y-"
    "Y!pMs5ojnyVB+wA8c5<d+XfY8C?prwy+RP0R!Tl}Px;AIU%%MRFNL<JZtdED$y{xv-yj~NT@ip(Ipu?<~xu+-Gfd~Y(kD-wBZVi<"
    "nYdA+?w~>J5^+F!r#N!904D{MU)P8cKOcap-"
    ";auodRcS^GO>p>&vpZr`7rmk4<0?hD6wc~9TnPZgEU>EFF^sJ6A$CC*k9}LnVqIpIEdHT+oFmGK_A_$X$5a(M>SWvEafo;h#Y;Fl"
    "1A9#~k-PK$Z%+5@?>uagSBp15F!5!e6Bw^t%#Zm>=5v0vy#x{H7w*2bnw^0htB3V~_sZQM$G-"
    "t%oMj1eNkQGHLHoy|rbgS(`>GGD>JqMjz_M$A+=TT(+zszc1<$!23(i{QNoLh0wBRLAh>bT0#}1h#yT&33B`A&$zpMd|87^+D<hU"
    "hkL28DpDC>NG0zZ4dfC10X646*a%UTFT1~v}Pb=i#D0DEa9=gwP$)Ptqmx?yMMDghQ3*<hNjoV9x#k{s6aR(GY`3%5KrC|oZo|7("
    "<GRLajL89C|kVs;#Qxn*gTh>Nt>9Xz&WwJ#y$PIQJB1v=rp!Ac~scmyEj&kw}&c@9N*DA}C%-"
    "%LAFAwJkiv|)e<i>&#S)glRQCWAUc>dZy97EG8fJi{bxGaC7fecse#f+N`dLcan7pG2lOrmX4(B`qUs@LRhQGX|)-"
    "S1F`{hvW3sTlqC0CGwh6)=BWevSk5#8_=A|ab>22Sj?(9#s)t=g*{b3UzJi@3UU^G1S8JIpT+(Ef@84EJ<VDg+8IjJ&r20=IR`Ww"
    "n?9D+<@6jN#tSfMW^y0+vt^N&#PppDd>Jww)|hcqD+8O#1*(IZuKzq`n_1(>dgPX9FOlbBK)P!c7?sDgi4IRBGc{cqy$j3?rkg>p"
    "n8W@hmJTJT#EL24P!=2(NAubw90@K$`6#$tnk>uw?dp-eGFEX=F2GOAIg9=P>JEo%6m|wkjfL^Ap|p>qvf=Ua$o2EZv{?pS{Xf_;"
    "nApob+$gipR}L9oJ_K3ogANcF8>tjWqlXn}(Xl~=EO=+>bkdfl#Xv?vzw%JA*#brp<|CydlLA52Eqxn;!{0ptWM>`N%pGRO*NZ_m"
    "lJ?r4<_-"
    "7LdjheeNi`dFrZ+BocWGog5DTI_<q*)%t*itO8#U~!R7bdmBf0GFvXc1?8Q(in5Ox%K+B#bNn1jn29fTn>%!*O;jC6FnboPL}W(s"
    "5vZ+a|K2rRIYWfL9awMho+mP=-"
    "Xr1=D6Y7|rWKxXxBUvE||TS{=PX@>w_VXF3^&$teIq>MPRjGms+M6cH>Z+k4kY$f$_4WQd+e5kJH*(G2mL3_mqBCn)&pqC<`5wv$"
    "N_VcyP5{5<m-=IjKza+YSmi`wTA~tS-"
    "N9wfXVtLkRabqATja=0IEHTHYCd?T;TBiX(3SK^D2V%puH{L}J5gRLkL2O06@Z}BK0+Er(W3Vdz9;<Gk*_sd?eF!`Ayu9NWFHckN"
    "e!OwYD*D<^0de85M$;p+GLz#rvgwN;I%}gBGDS8EGsro)AAl(GY!}jU6K?;Uy{R=R{&vX+Y1_PYbX6b0ZcpRYm<x^eGb{D-"
    "4+H5d8QaZ17?7CA6ZyMwA%qgZJ6q8(0J}C63q<tV#Y~dDHUzlvsB(3WEf(PNHm}C{31=qjgr<3@Kj!hYfeqN!I=}T<-"
    "fTur1K#|x<bea0m#F`&Xg0_f)y6PiICCj_L)iaz;w#D>Gd??Kwp??<l#qnRBzx{KPNf#II))V_s3tNJEt8nYw52s<i2FSo(GH?X)"
    "`K?dV~YYUX#B*!4fSu!R?8B65fe>-d8dp)h}-"
    "~ET&5m<1s{UmzFRj}?zG1|S|riL+ghYTxyPymuN7>+Fyi)VqYq8Ma<m>vd9<pRnMczf8WF6D=GfGOZPE7GuL=UW<bgCq4148w0c("
    "ja0648AB?B`*SkJ$fH%cxG;rKS?tIH`ODZdx2YWTN|7_m-dwi~g(d0Qep7a-"
    "7;UOBK6LE(p)ez^$C+t<Ur1Q&>eW$^@ry9tda>_S_p!jZFYn}w8_ib4!jhUac;{VZ&0ZkL5)anx`D`@CG4gj=I!qAp+>93U@BOyw"
    "7^;`10UA(H1!Gia@7Fn26oB)TvHUS44GYY**>M0U$Ac3?ahD_Aef70Tvo88Ec}Aci;<U-"
    "XMjjiYa^`QHN==K|uEl_C*4z$~u_FUmcJ)z|}EyI7X0&m7P*?IQc-)?D~yVP3Jw-MPL2XGk1T1BOjJ+}t;U?X&f3{Pt!PW--8;1<"
    "|k)Y=q|^1#`mijE|q!ik=Kbtck$g*&$(;%oeIBus(Q2EeK*7bG?Fy<IrbqE7SA6UKa)=n+`aPO_@c)2e`K^W({(GX5QD}^El+#9T"
    "SRNPV~uIT>4Y<Nwi!@(e=4=F<ET8!3Y_P!E=5qV=$32qYo$EUOOs45@{M(R9*x=T19X~a~f*o84DTd_mJ*K&;g*Z>nW7;{rlRCrd"
    "f74bKlW|@5LsK;g!Ze$~ENPXWw?v_+XDf4&drzT%OSY!r?Yxjb_!wm>JD>xS_wg;VlE{7GoE}-"
    "CQ+W#1xSo2s`K0kS__d7{Eaq_KyQP*9W;?uN}Mw%&pPhV6oOyZP`s~gR^5B)TAMp|AwK+XNhGRieT8nlZE=#Yk*1iTqX(5lFGpfV"
    "#cSOPEZXL5<mgz0*GZeYP$Y8Fv)=kb0H3!(wLj#HG-Td!5?Is6kM3J<yvSYGjt%NgB3iBj?cE$>BA0Rxj=)0__K#IyPa<?t6-"
    "=I50Vu3O2rOM&pd0q#<F!hgoMXj2qH^rUrv~p(W`&^x22f&8!sN0FtaGYki2_{7gr4uAY2od=?1@YxAu{LCnU>12;IAgArUZ4I3x"
    "SFJ<LwQG=o+=A3PlIX0u<qkqATdl8l@IJyZG(AOI?i-"
    "U}nb@_2(Z^RQ4_U+y`^P47H2Hg7@_M62Pg^rlv~vNORviCS>{Ol0hCzrD$uNnS`MucwkI0{hC6l-bxdR6w#%W-"
    "vM_)Wb}H5KSbWtW4$|0AISJTBE@L)6|JCl?nryDjwUY6H<7`1KoUXqz-"
    "A?MmdO`0C$mRSEhV~T<;SHc&|N1roq6=gcVx6bo~2VHVp%rX@m+D{Mak)D?Zd^=jC$}Z8oP&uz>alB#m8_MX|DtGPocW`mA-"
    "%>|lj*Hho>9Qo-n-na!|{uat5eDX)ygpmWx)SvSp!SKj%V)fkHaZ9IPf1fl>p<)zz)09Hz^tb<ShIlw&DmR1F<Q3bbtUQLmz7dwo"
    "!;3(z%!-$MMi6EdPOKaGS2$=dH6H1ZQ8fHonrd~2MtjU8;hB?7Vbw2s|nyyTjlU5RR42$Q?UdgY9iB$E^-"
    "H}^`pE(X5Md;7y@ljc)A341?+(cY!3TgJv#?Db@ok`-"
    "5Fz&Gy!Ig9tnhz++Vj$LOa163A0~DqWfPA`3J_uxBp(}`Jk}r`{LE45P7{*x81(vJr7VT1=Qsz9g{fu1R`V#}tg5dMIf1?Q8mD#?"
    "@sPDaAAO-"
    "v<;LCZ#?G(BNWC`jE4^CrBMZHB&$0n5o<0vk+2dN{h)`E<M5D|nThi0g8O>RcF9Lw6ERWfd~vIn1|Xon~b!^v^hZWGERSZc7WAS4"
    "&e*`Pyn1Ge{ORAqxj@`4@GTC%t)9S`fh(3GVF9Z4Sq2*L+QN?pzCS;@B6<K;&uX{k^_-"
    "<dRzZo@SXcSGrCeh9z&9~N{02H7i?e4lPyzHnxTD6h;C3iY)3W>?S0;3&L*DQlZoC9gz(7yu%T#0<#7@&d?UHX;2dHf4{)6SM~><"
    "JpzH6r3$aaD?olq5vAfY{rwW_FrB=8lbBU<u-"
    "p01OmP`pa0c3n>j1g5ih%b<tfc0;$seb{e&f`!#dhDkGM~n?_)PW8AOrW;*<kb&`mtkBN9%+inH5bXd_zAXrg}yOP;4zwFrVZc5f"
    "aJ8-lULR!WK(N|dsM!~gl&6>6j5sQvDS`-aiam$qHxSVJ$XrfSEEYT%2<1NjZ?J-"
    "6U9gWqQPKFy4$={jvxp2Gr$HRU~*MPZ?1hg#UiUz$-*m&u%4lSC8C$7B<O4!7CSFL}F`*B9-"
    "S3Vb$HvA`MTS~#hHURZ5sAV|^uXL5k4I~$izt`)VW+>79GN<fE}H6quFOtir8N3DIsG@WvkX-LK-"
    "Y{nQ4pa}ea?$14}$}mC7E5~_clxNOQYIEuFknelYle7TK3BX4|AQfs$BnN%Qeu_M{sBRB4OS~~WmA~n}fe;bpA{EiB(htal!g^7B"
    "r6%U{tKBe%W)WJ0m;sUyj*UTRuxUERGDPl78L*vEWg<+iy$0YQkKZ(FVZ~>to(cr^TTvunbnt2h3i`|}Y(*b-<^}jKWjH7dV-"
    "1?6Ufw9zYO8BI`EO?mr^N+dfgJdaB+fyVw;?9ytN;9M@tP;Ic~NT=>-"
    "2)&MYoak{R&~Wb2b}{q*seoCY*by4Fzqv6hfxQuEyFH#PU?e6mG(bS$yoD%DU%WtbV&f0iK|g{(Z_}e;J3MF!PNoF}?WJOyDn^Yf"
    "PMmlI=}sY3pL@LpS?jrJFhgL^f}IrbuPTCrCkwyr`mY7Q29!-GUL6x5anG-"
    "}bm<{c<#64`#L%pp(2g5rxSE9hwD2Ckc!o*36bHwk^X1IKHevMuM9L){E^XK#@2n-"
    "t&XfZR*&VjhEYrEx#Bn(Uy#aSWM>F*sO+fBLfioidn};!xrfs^)0jo{E$n&8}^v6Y6(HNdJd?(zLC5A0@$KQ#Yx1qRD5Y%-"
    "!K8eU?;huP?L@GO>->D^cX2F@&-@xY8eUq5iGGR5-"
    "2jnzDBX}lR4S0;}nFhgIw~nCADzsnry=+m_$B6>JZi&5^QBt$L1$5P`tn8@+?`T&t1r4QyhmfeOLkIoe;SVZT4KBi!AxaVrDlSa3"
    "8rTI$stI0GK_?Ski|K7T;$NgAog^fgGnTwi~%5v6-+JIM7e%7IW1y5?xC>lD-"
    "G`(A_K<kXo%Qj==7Zch0>OLb70(t#K@$DHz?;Ck7tKwR1%k{VI|gs#Xx1L{*`(kmGebPR{(-Fe*<2Lu-?0Fztv#4{k_l1F^iDi{-"
    "bajf8_vp9HfCaHKZagHk=ueH0B!w!*rAL2(0VpbWoZ275=H4WsggFo!UChq5<cHdGe=d7xLx(_^VE;~6ymX~#%;1$ma7{Qm5)$Ct"
    "aoaR=5aLu;JwsBPx?S>~C+i@x9;uR-~(IAxC!wfjsWGPpnQ8JM5-"
    "v_Ei?=81<itYdidx7P!z8I<oZg$o!7Zzw*lt&CUg5z^ZLRbj1gSy$L#9~5A_?9Qf)@46_EaBtEpng5R1fCdgtCT;dV{p~Bph8-"
    "4cb99Q9&V^=`E8`OzvgJ}%?2K$%#c8eq!L_328I1kSHhQ2>JhQKwT09wFgsyW-PUmIo!((ET)M~*1a#UOoFMbZah#W{pU?OAs@LQ"
    "ws!i553<FCXB&~)@3tXHqI;$`D}V9?gHGrBkWPLpS01|&I!8NCsw{gc4QBI@mC&C9)FZr!g-VBXb_U*36so`-Qg(;k8x@Y(&o-7-"
    "|8IU#F<2~hMU+7CU<72v=VC+}$U785${4PPK19-"
    "ImL_CWlt^+Q~c>E*IDk86tC&^UbFzx1j#Hx`dmC~CWwJzIYV8xpS4y{W^9=>g;DuME1}GWjTYN}jOs4F7-"
    "f_RLk$IK@JVWR8G5r>6V1L71a#+hfX$DHBb>Kj2N%;0CNGn%V#@A-^myYFv0a2IKd_+iEt!#GWlU-ON$Bb{n^vf}6Z_;o1bl2t!O"
    "Ar&;ym{I;Ea7PPZXl7?bgw3&9m#1XiPL3fT}v=?Zy{GxwFD}=0o%gXjdKDVW@5g2yw^=PYFGzeFoCl9;I)xvRh+^}>Gntyt*Z7Wc"
    "2i^b@XhM7RX)-k{5O9{(nUPXk$yd)zYW%qD>g&-iMZOrBw0dJl_(1dJ*NNUGBAvYA#Q7qe1o+=f+au-"
    "D?#VTsx%qV)cxIJUwTh=nP03hV-!JI!TsfR+-kO2D1MCzWG@uIS^SR^`k-Y;e~$3(VZ8R34$-CBl-qbmaW@o=-"
    "XDoI>PyYLv#^uKb9q<Tfpy{n`(Q)r=`g%18DhYOc0M2|7<Z)xe~J#|JmKnbc@CIz6b!u_J>&bA&NQ^*M50wn#2_5x2!woIK0bF@="
    "n>6Z4DHFLM92K)`<*p^&BmOpp@Pna{!+Id8E1(1Wa=<T&8(f+q|EZ|DF2FspR3~H3vk^oj6Fei2AN`IN*RRhem&1)*f9XtRml3D%"
    "oYK*}xJD$qJ{0;I+V|`!e8gKhf??00xbbakBcT5U80FdiWv)5`v4i}Cw!?sGlsSM^<@I<ko2o~IDB8g3`UQRzl1lMhtk^L~|9rFZ"
    "w7pBLhoZ8n0%~bROYeP{Uc6fo4pC%*B8|+)!A?J*iP$p&&$5sArHOz1-"
    "XB*@63@BS<_f&8pZ3vRpKa}!AVfz(F61x)Y3GF+}+UP{P$+O;B0VJ0l1vjO8gYV0=Cz#uBH^sO&{OE4jLoG@sh#@)kU}5tLF!J<T"
    "_(F5>Fo7rP<s-5XqYZPBeiCpJGSz$co`qU>7y4irYuLj@wrp_#s1mTG->HkU_zphIv2Y;Fizwp4<MNfYCxxqx&X8E|9_44J-"
    "?K<7YY*6bh%|J3?0yl7>5qore$wnHD^3wTh0%bM!%S~&fD#K-x&uzewA?a&FZ(Fl5@rb3rY9J07Oeu-lUG0NpYsmE-"
    "j2<o&(2039(2gRy;Ql_7BBp-0I%&-Tp0DS>YCl6J=-"
    "B1V(6D(1=$Ym2h?$9H>klW6cq&>1X32uuX0v%)Ol>J23s#r9gm2ZOJWYQiWLj)=7r?PB>f^pTN}f$&tuQB%VF40^~*r1r73<3yVS"
    "=izq!>EF$=Botep&>iOS)Pvh78Z_Y!W>afHnM0^}8Y^4O%-3mP-QtmP$MwEm+u|5A_v|JYf+;3FJ2*x6m}DVB~XWO?w+R#xigNua"
    ">WXu@yaX)X~{QwY4p+ks5Rb`4O(KR#uE&7W<3nMp(@fT%{hf#WIMfV3ugHtV^JQ)hw89CdnW^ms|wo47mzAsqX$1$0Gyg?A!A*_Z"
    "8Gr>uI-OG#P*c1W36gv36v%y+i!Z4s)*O2qEtYg?j)qgv`7Ih^>U0*AzTe@r<xH6+xbS5$TQTu5R&U3|co8=EQSqLcEyw0)qYEDk"
    "d73HRH?b@XNk<HU`0s6D9CUbouZAvafYBkh7GbnH5poiO|~Ab-na!X3%mH&-~Nz9Lv4An7ju5#3i-Y2H3aai(7wdOcfvd-Ti_m&A"
    "(LROf|S#HALJIgsye|3#!FIgCS&1%nsf0<-"
    "f`^L}bjxIUBh$iH8N3Wui(*E1_}sz}ykfQuy27u6#&W?XzuYPalX#ad`2#$)D9;eD}(K+XNi^-"
    "6ecq7w!n^c*qx8Nh&0r{Pg0sA}AEBClBOx5(qzVgpU&(g{eyD@*SXn-"
    "15KCl|_r%2Pb?Dbq{Inw@JaQ49`4{c^WmJlKJd6*07|MF+mkd(RQ9Nx9~U17W3`H~xD55mbY<%7Q!Jywz;H#8WN}1MLl#x4Hgx8j"
    "+_=qrFakvnTcNaoG9!+dYjjp&6Cn|7rA2(6A5@5l2SGYp-"
    ">C5zuc|5I@WTns*5G)RSXA%G3*@&Nu5e!=ka228Mnwvf(p4eZ5LMCC}bp?=3AqM!caRUb!P{ZB_EtG^&tn4sz8e6B^)p>o1dW(zq"
    "wNe!69MKqYOAjV>{k$8w)|3XBF@rdT)|Al|5;UL<FrJ>K}X6LeYmf*HVerW{E*2jRoYYy7MtgxMyV!^b<oXC<;b6CIGCZ4bq)qbc"
    "ICYGoT_%fg^xay+*Q^Rx7@@C4yP98UTxZXI(PCXke_s}@n=X3%Zxd7{j^wv~h`&fw?dVtQBQ-HYXGMszJCXGIh!Ol;Y~6%2J2W6+"
    "sejL&|2-IO(?Y<Ulv-"
    "$GKvhAa*1nL&_*fq6#y)w5JfKXZ8*=+A_LSp<v^5S?+00m?wh<4CWeC=(8^Wk5We2(QhBup%X3{0x}jS{@Ivu2+vG6nx+$x^sL9x"
    "?v`U^xT&3x11e~)_%UBnJjJd(p*)*UI`oBXKL|8dQgJ!4UBFFYaHHl>8<_)f;<peb{ym~9pvN}ZWcxeH8KT~0{g~z<Fw)%Gc;HsU"
    "DDxuML+qj&3g_zYSL?XL7sy2>6=p~yx6*;I)~cVTUu{GhO>%xja+koy&idb#d^w=`5L@yE+-"
    "p`0!AjI5fO<>A@{W16DD6=os;}$76~wxWkNtN5vEGin?wfU^~iu`<*4<N@8|Xco!J|5JdsG=AGr7Bk~o)?B+ZC}mkxARloWocm&H"
    "DNkDGbb8KR+$G>XFShtndBK2~jzm=bzh9IfNcuwSuExSS6i7MN=-8Cb}NryCKbvaO;l^I{v+-"
    "xag0AWnmzbT+$Y0tABZre3$&2=kfTPNPS;S6wzvL+ru?vN-"
    "wNkFx;H13uHaZq$}Y&<>gQy<`#(J7Qp-0!Jv7&P>cq+JI$Mj99NZhS|fTnm-aJDg?-<+Z#@^ZelyM7a-"
    "fZJ^?X&OqcD#dw5x006(TWMmBdX$?%e5VpQKnA4E4y^GL%~iYGQ{JhRM6aS5X0&1U-"
    "dV*DCca=C^^E`h`?)eVzU0LU8Qc1yL%w6q-"
    "zK9@u5UC{wB@L?cQYtgqW9`DIJ$=5SE)wtraCO_T#hSzdn7kU%J$!Za@%=UKUUIr1^STfg!j@(j7XQ#;RWlR7Wx8{YwO2;5Bj%?|"
    "`yeN#W%^XuDYpM^-"
    "QvSE1xuTLCv1D~JGg@UI>o7sbn9j9?S>W8}O)R0n9NwF@Vw*_L>`0;rhuEw*5|F|A(<yW^x=9O2zQObF!R93G09ed)T;|b4<hekS"
    "Wvw$V=?VMTMCn{RRwSH4SliPTAOnI-!3^h|ClIRsRv$yfzzE<3=O^<CT4#lfXMSn^b~9~wJ^Yk1{5%-"
    "M9Ix9+FT>%UjD>$*%v;9M&npbO<*OrCnA$MrvWC<?O@G)iPy+dxd*#3f_YzkCIB91QXBZuq5O!+)j1aKP^^^=fL$0f8p(w!viT$|"
    "w;oXq9zJZ05l`k|$r1XH6U-"
    "_wFglkhwWQSpzgM8&K6TRt@e~yC9MzCato~FjjLPh+WO+?Cr>OBW=S6U59dbDj^A}5>R6xp+gdFdaWG%K-"
    "bd@fGAH7Ve{fbkr2BxiU{GQ9v}x?W{n)frRL2D6CiL;7o+SZHti^zg>PQv~N;nAU{D+$0D<K#J;~yM;|EGgSaH^is*ofH}H(yk6v"
    "3Y(CT15BA-3_eI)75zOUhviU_NW>-3tCwG~flPks@@Q@U|;_UHIH^-"
    "3P@hdqFf+(ED25x0~R<$>EvRQ@R)uu&xSF9R0@dnq?T4l%<65ea$B;?dmPhmN>Qr6#cS@&W=`pz-kKNLmq<GE9EY;!%!Rm0xCkc@"
    "g15^&kkk&s#XDCp*T;o2~HwYIEsoB>MAiTG`&!q?B6tk*6y)r2!lr)*#S)ow?s8Y33lt6<X!2=lS)lNn{XbMtT-"
    "=tY8KxMjeOoqBKA%)HxDrN<7ibYu6B4}wjry=Mf%FAVc>UkMm^qnGL^m!}u3Bs+{^1yhA(CPpnX#L&cZNf>3xU*Tjiu;Rjux&kiz"
    "kEK;#^5iih!>;+@5!SQZ;6~A4GLK2}9}#a^(a3Niz-Cxv1RVW(^gnq`FV@#Y32#0Ri%!~-"
    "$AFIp;>)UBUlOe08O$0XAmj2e7&3=x))}JoKi&xktP_)*Ztd?07raqp4wLc8k+{ZH;$3ib=p|>f+d&E9K;IB|L+7aN=-"
    "5fJj|*W^IVic+1}zys3pb86BzNZ1{J)wcLEU80q&E%5T^;6yEXq<F?uY#=9vQ&s;OIApMi@;{BYDNgmx?dUi|yLXhO0Yk;kAt^pt"
    "&==+`moI;16h;_RF&wTUkAX;!_2{1i$DrwC?q)S%|Q4+P+j?D4i^!(8NLiGI;&h-"
    "0lI|jB}z5lOu}bs)}&})1OxbE;L#uTLY{90Jd2xflDD1z@YIF!MA5Vo4s*q+)TB8sl67DFenVnbD!x-"
    "yVh7SYdM}|EStXEG|$I0ikJO|VQcez%Z=IolF=2^?(^CKwxB1rY<)HhH&_mF-InrGSj&Jfrms(~`GTP_V##FBHP_I@z`L<<O=Oqt"
    "E;!NJajpetZh509teAu8ZUk|>6c)6TZS%cI<g8O-Ev}t>vkE}ax`q!FuRBlLv6>wO-dl5%w%%OpvJz(P_tQiUn-"
    "X7!;7J8Q|AnXO@CDdqs0sdTYxeVHfu|<}=*GzP+xfNa?aY`RbT*gZf<)rcNr9I}w5xf6!vKtAVR-"
    "!(`l#N$W97@cWMe*C{XU!_b15}TUu_>-"
    "W}57s)AyiI;xi`Hdt6*T+()qA2yhbPa~zr%1f&x_4^##g9iHWt#A&>^Y|#zK?7dSdxCw4s_nOdL_+k|(EeVqLoI4}>yzphv53Qt9"
    "BCUc<J<L+e5`y&0`({s3OKWE~b4;p&&gqETz{(_M<;$COP7ZGw%A)jc-r-i^oMMc)Ty5Ef+;}m-"
    "zy@Jt%G%zvgdhxlUe<BAsgR#+KT!)Bo!1wdAVg0&f~V``Hz7o<Rb0y5D1;a-QdyOoi(mI=rP9uB0PPj?{)MJ(W)*0r-"
    "t!{bQ41$Lj6h3l)6iI?fN7-HKI;EuKYFQNSxM=|0y%r6U4IZ`!0uPAMH(wGYKpl6KfG1~moK;yEvZe-"
    "u7Vqe!H=g3{T$z<VG?@jB0jGf^oPNqAQ4nlI&P<H57!4%!rNlEp~T#L%>;U-"
    "jo8`=n8SeER``H>#7KgF{w!F4+>$%(q0_^=L!E!i{sS){bRKAha}z06z(mFE4dJVZKCb2wE;BMLu$6AjYQ1I4khpr#;gR4T+%O6V"
    "XQVg07sA+1x(E&y=v<yPKzEWKX}G*N9PMOT{%zFzj0>42=iUYMp{SIvod{Dbi{zA6b=&+go||pP*cjkl%F#Zu)`spCvU?6HVB~bX"
    "%<%u8wc(ELWE%nbTzXksNQ;Z5Ll5_++~>|SqtC;Ge3i>QIo_U*7CNQ7^07V?65@<RYZ>N*A&T2&a*2o|?--b$rEoKI_M!{(%Hv?t"
    "4LiFn)4yM4HiO$(%6b!U8#Mr+KA=VEISIuXK!0&rsoTm<6xW3F2=oBzCqg${S3O6;CAm~xb~$8cFvv@vr*`7l&xPN(;FPCB%5s>!"
    "4h84^+#K5qACL79v{s<p$9}6A(!(!%ggFoS;_A@wF28Od#aUDyc)j?>K_p7Fi`fH97-$N^z}z02xT1rz(Q#!BW!@!Y3--d)m)VTn"
    "0e*%VT4lCmOK@xfK)FXk8We?kKY*$|CqB-!5Y4T$U&-"
    "+^Po7qFxNj*#5qL!d^*eg{?C$ld?GJ&2dlIWm%fB{}0mCdv61!r2x0<(+q;wQ|CS@%qkuUcT08(GOI`3nxeA+fbnT1mFYMP0<2C~"
    "8=*#zK)JoI@v&g9OdlPaDVaETf3BoE0@^|CJLt-"
    "?u;EzY+MpwBKZMXFv@%MjW#U!P#HHqXAlT5Qza@Zf6rl23+tFSs5W$^0#_NIJd?3?Wup1CrOMkMTVzl!I(y-z{%|iGaZ-"
    "6+kTWcepOOcmVkz!#NQ88w{WZe~=x~1ix7f!r@Ug(6eQ$*@-SvI3EcWWF}^IX25oJXY}hVo_E$VFvNExHxLgmswZ3TbOu1D#-"
    "!it&=UZL>8@H)1u|XKDkt3@qUz~YQ9+%2W}_^a+f+HlGdlJ^nx@-"
    "F@qx4l;uGCEEc=?NfHpv>#L4IV)EP=b&pV0wSIRW~t!w0n2~%nN5v{Op=h)`0tZI?4U}$Jp#{e)uuLi_B1byj($~fcMnU5{Ytu}O"
    "?=wRT~jjBL_=^1`7vo9U!!=3WlVPU=ktau9WETL_DWj72d0C~7W#7=<*{!GT^sP1n(_u<(R<U>lssJlpF@YiVwa9=3&0fQ$(27kN"
    "d@YwJ+G8l-8`FeRs&zp+u>R79EST)OFrrD0atQ(qXHssh>W@B#nLk;S-"
    "pj_)%GDsuuq@6588_e;BVa#Sa#D33Ptg$?snE~K0Ej}kLMw}0hoUOfZ^hxT+V@W5!mXw@3k#K5bH-jGV%r>|MzzL~nVjEMN93D1o"
    "z!hxn>qTl`c4ReMJr4pNdQckBvDJA++^aBRcK`c%&|r1+=u+__eqRpL`QNR1L9d=OY5J)NCmWaj^uh3f=eGj`=O_S99x^Q|a25}T"
    "t>`)76mGA?Do#Cgg3qD<E&62&pA{<j{d3wI&p4g;Q_$PTM90HcFE*~(KGpGM4?5V+thE-NFS1A&mzJHvM?{dg4FDixwv-fW{DoKp"
    "hpoWLwbABSq<Vj4I$*&bs0M3l(*C=fsnB;=-"
    "0d_Zr8wj>H!os~{vEm*7nzaADIp~lzCAAyWX2Vx)3Z&^s(B13nhotxRHT09e)fc+^~U%Aw_#?Pk7-qUGE11HFx-"
    "Z?z`~vQG!WHLG4Aoh?C~z3xujl(LX(!Mj7vKj0?Xpd*QIwalz86Oy?--B-1xS)6y~Auj-L-"
    "D8@7;1_40IvMG?$0a)CA`wJrAQj>JZ0teM28Z+IPGYFCE~F@;%EBcm5M-"
    "nNYo#kX?@gJ&A__Q0wsn`Jyo8xBg5oi!^;Y!Ipd4ib_ab-"
    "u=f(O^ka?YhFilkG3$|Jsko^h}Q)7r@_6gbA{Tl>*M&EB%aTYzn4>%RqxV*Bj61nxF;!uoVM8Fjor<M>P3i2?9FMQ&G4>ufHtR&H"
    "9D`Aq7AB@%%NYZs@Q+?_9m@_aXp7C~Yiz$a6}qMAh&84I{x7mL7mYj=6(WiARg=04#kDmu%xD%MdmdT8&spVz2{w`lBS5_Nrtr>a"
    "5K;3}8%L!G$+HjTn1lXsZ5%_`Y*B29*BL>Yt)DHsgFQ;<i_Shm#+21a5LSeJ*Qp{3A9HxGwfi@-"
    "k4QU3*~+k6Oa_+3g*nG6}NqFl<+CSjG0B+}#^XRUleoqn_sg#~EgA4Das<SQB&yrax>6Fstl6AymUqqZKH^3xjv^2vW`>Fyd=&v&"
    "?H|Q0@b>Nk_0wRsq~bKjARI97$_mR9UdJw}tU588ppSSx{?QuJ`8Uu&#+OVq!oIMd&s`v)PZ4<%5@I<7d?n1m!-"
    "6?KK740qf^a&V+n<nDk6#p1HuD`W!6}XkM@NFSZtiM*z<=D*$RE)7IHfnW?BDh2PfKkCM4jEDmdidwLeIs)?yC0MXBcfeXU#p%A-"
    "C>$2$eSVhI`f2VjB_m$%c<t6?AF@be!gDEzgp_l-m^Z**-t>x<aV<)HP-"
    "2x{K1nmgZp++)2jo$+AkZVqbl9`S0W6c`d9gCkrt&iC3u7C`ZDB(S3Cw8^>pQ6!~^Ygax^ADRoy6NWKHtFjR)QV8!a<vQRXx?t}Y"
    "#{o|**gg9ta3$!`2C>^j%JBO5wTn?8mz^3s9ZYeLZD*e_ULVf8KQ=#Rc+b$q7o#2;ZVPs<`zujK#ge8^VpjhrTx~>abxZjjTu$K#"
    "=3eSaZc^V;{6_uAy+<dH6)E-"
    "oTv7ixlp%{3sWQDpe#S^BZB)aTOyNnI+)=vGw52Px*PhC4V|cJY9&vz<P6}HtdLYbvy`dpXe_e0voC~M&a7Q7<6xJ%sO%@4+fxi)"
    "9JG;9mkNx1O?b<BQ&7^tfSkQHq_>(CBk2O`hr7f{O3mqIAso}{w$eCi(*vOgw=H)Xp?QOIv#V~p>7#1?yk3VoHH`w;<?2v-"
    "uG2}Vw2wlM1x1iyc8zenwF>SYn*j1ivZ#<~%hHeB=gR63if#mLbsl%OQQfq+$Ipn|W~pRqhn$qtAP(|E8v6?enBo@-hy`OH$_x-"
    "l@4Tn6ig^N{hK1FzwRZBazdT_4fo)GC)zb|C*<f!OrkHGRU?@P?J2C0+svMXjGyjCNk*AJ|LT?7A`jbd0%a0LLG_aj>_8%F`un~7"
    "lCcO7IP%yDOePk>(AC}62PWJxMMA|ixuKZar5UYzIB3<`GsdWa+#P*Xh<|~lOHX6F_fGSXSie^K^-"
    "B$3EOn1(n+@pCIUa1M;IM&B(rbKd3wuH35_>*=o)GdQvvwNXOok9<uUHKr1EE4=q^DwBH7}}x%=tbc&8OaP6)}vNik8tI;OgfQP&"
    "^bC*C<b8cDXDa_a>T4H&upqQHsp*Dln9iCXrx&bx6{RjF6r1skdFD)ki)|flB4ODDF@}JW6g!JZ6+9Fxpac)t!q&m#t7g_R5fh4C"
    "_^3(y<yI`EOUmJA-Jl7z*c)lsZ24gvM16j*CyEP2%RY+=1&Cf1%>Dw+pA$VtN;Z2Fa%32*u2RJVWGXeYJFP_?Hqzz?bc(T77RcBz"
    "hcB0N~4&Bl2EY{E-^J0z1^5GM|$3P%c@G7%9I417Nqd67ViRIk<Tb4W^RE5*}))0YPCCIKQpNZRqPFTyxA*WanY-klM!U+V>o1t7"
    "^{lJgw5;Jc8`c^YL0mf6wi5cDTX=Flpj5fA)wFO6V+a3>!%oHISl*;_LL~Bk7Sp8wt!#>*fSNgZaS<Lg+)Z=*{s`gfk8=)$J18%#"
    "H4mJ7MSI;JDCqNTSbMOggkgDR0MEGW)g@Yi%o<yRJ>&{f=Kbt%E;uX;)j6+kfFgin8)^xy!Kd$9OVmoF_a>10AqQHw@MHQDPYoGn"
    ";{3r#aMFO96CZxx24pwaxHEOG()yx;An*aH3#@`tB*~?R?=BX^DW(Bzd~yDbO^AhAlaIl0~%KBIm&#7*~=n{XS+-"
    "co4qs|IL1OZ<`}=SD!1AV4hz${a>ir}1bKk%5GC1Ww-"
    "X#LuUcuPL7@KT*Rj{Is6*@rU=7<!PvS%l(NU4s80LP7rgWqsY!xIf)s{nihhhGJ@gz5F+6jb_Ct}%5r(nA{87$|{z0!--Wnk=D{A"
    "%1r8s&hakI<<On9nPrt?@*`m}5Ow7xM>ycKN{iy2Rzz?bsYI4Ie0Ie|8c~$Q@xN%SgPr@+4kFP_vAi)+z*jR&aIqR)u`?qoWRFc="
    "~?AF!~_0YRqG6x#{)#51X6aB$3cMF<7uyZ&|+$n?bwn(D5wsD+B^qA)r7bS#8i6c!m~s*H&Wk7={ksnALOp!!%ysZG=`y7S-"
    "Ffw#GgeBa`g%dBaP1So{>5|45ORq$6v}9vhb~xKx~@C|?JAiMthktJqPtl3;7q^YAw8Iay%DOb<uQTu0um(S@KfU|9t#1qaFpEjA"
    "`1xq=(dZAg^pxa0YN$4)Q&#N0b?2~HE<TnByN^=yqWl!&^o{Y0Sv%-5<Q(9-eS1%i<2+Pi1A!Xe=-"
    "ZEux4uOKIcI^dts)>h4Xp=n}*jvDq$qeeRjFo2!5RgD8pONA;9Hd8?=Skf~o?S>=sM*@-"
    "{G=bBh*BnJPnjvPtFnZwZhDsKt6*xvMNOyW19=}jhD@jGXV_r*;#6qtYwGdSvBv&?E#~%W)0MdE=)||QLaoLj#hxJoaq(sr#=m@&"
    "@O5jH!Wz%4$gC{|2@O+>XwoR_&mY{|t0`FPoK9h#eJAgT|Ns`B2yc?(O=`cs#!eLxCRB4soSQ8W^qyb+(|NVMiK(9zh{gPi1#=Q3"
    "i8az&-=4XQhbc_pg@Gz_eIEJl)H3DU447pu5=onJfmp`8E9F7%%4CXR~d8cIKjX=Wq&2S-"
    "iQVY?=<*z@!T&6MDr6a>C)h?J0o8Wf9%!DSwSh9GNk4h8NT_L(pbQw&*SOAw07IPnliENZq%$#moH(We#mlrOzdXu*B8!fzKYxOG"
    "2tng5Q2R@7ViPQl!9{a^FiEG^}QS5$>4^ICvEtzw_TE3A?0G=GGCN~_}kD>u&boBpGb#J-0WXo};`@-"
    "iW5%)`VMc_*TYC}B;>U+)4+&xxOpakli%)Qr|Ghz(CnEgf-)khx74I{PPDl115%*nbm-rD4TO(7-"
    "8n34^G{`YBSZLNOane%8=h@SH`q@3sY%d=915m3VrwTIMY>R~)`D!p~)61Z+TO<*eSfn}=845g2VT}Ov#q7YrR&SxqPW~o(A)Ty5"
    "<GuTyld#Jv=fnf#89ZMqSkgHW)K`gBu-4^WUnzjA&iDurUK`y~u?X8@P^!dP-JMw&8y)rQTxf9H5x7~wWBvti$QY8C;k}qVT&;Iw"
    "+mzL$pnv^2u>pHN>Z6XJj?Q+aGpLWsw<oB?D%QAPaq8H!i_iv0X+8BYA0xx9m;icvb|2#>iTMtANHXm%?diUqMF12DLTG=K|DI#$"
    ";PvUrzvx(fIqfQ73+2srlU{537QODKh$g0EpN`^cd^Yqo2QoucJPC{0+|5A0oTkV7V%!*mPe;aYHeBA~w(o!Z^MeAy|apbTW3zlb"
    "uB-Fxe-CBdB{7-RtGFjHusiH7wB(cBxDn4mCm4}HxdnW_Ja>A^0sTc-Nm6m6--"
    "rm<1ydDnBP$;%&Cc4ev@5uFeXl9CQMV4;p2${b_CeK8}Tbiy@GS)6g1`sjAWDi-atl=>j9<?vzNzB9TCGQ1+r^LaoO8X=)T(dGAZ"
    "g*s_2uv-"
    ")g#qH)mJTz9!L<S75jqFv3m7S`i5QFYY|sO~mf!oiEfeEd@L0z;*TOR8F4a~xO?QG9CojTz%gSI9)0quvpBFJH?)kQ_yqHtLkv;|"
    "VqY2{^dLo@Vu0olt!LbvvktmMION^&QFp=ZZCllflJ&h*q>R*SHPBaa&yTJRFR-kpej)^9o-"
    "bmUCGHpD8Q=F6Kq#mbJB_q18L5@+__+ycvO0TQ3bds2uEse9x@q|KuqJg%kGc-"
    "!8R8rNgEri+ty{1N~{u1m9qi?4==}&i>0l1o<Z$sG%r^S(&Z2GmC#M;}vqMSZJh!w7s7gGDl!#=`rz=%oYNU6wpvE$mh!tg5{ZPp"
    "JZV%R&~RTFS#YfBQge&$z2mE!#O`aHF7*vND;0MGd{%L=g__M8;96BZv)K|=BNwPW54e+?bfpzAS^j3nd*cP1?6?GMu5xcXwxZCo"
    "z!QD?~)o}Ok$w#SN9rsff}=Uy}GrJaAS?y$F1Na5ywm}IrVYF%f2zIE&zhQ^kijAo(vCU+=syeCTwPa-"
    "#1o`iWFO~WF0Ct`RSTmQk%??_&+U_<68v6qZf{>lY;OpmVOUMV1YYRmck2LE##uypz37+3Prr#x-"
    "HHY<?S@B}o_0w&oANMmH2WTjoAqSUOBGf$|NI>>J<Lm+;dj&``>;f&LMmTTCjVQ*l1%vY2RsX;LG&f4{{o+Y+Vw7Q0k9$6@9iSyT"
    "&;^0^l>bGsnS4VG=sUPV8$Pt@e=^=mf@xFb~uZRd&7Pso58t8~Mee=6Kvk3?YGY-"
    "o$2}Cu7(4EWTBz*&>tWNWWpG<J=?R%aJ&iV!O-"
    "qXH}bN9(IM+x(^O;Mop6Wxn%q*{uMG?O(bz~=2uVX{3jb>VNC7ipCwcrsv$X3QkbQGJ#0y)Sr1Cl=QMcXZto;C_mU-Aqx%eDq62G"
    "@Kq#683CK4WtqXaW!ehVp&sm<nHX7o!ZsBBU*;pSto>)P7`scLNhBZ2#4mb_}r<7_Dnz1t8UNdaidpWl8Yi9rDbrZDDTjdT=W05T"
    "Vss^P~lmfrRArw1U2IGXWDs_6};au8@7lZ3h~zLCiSgO4z5)!>n*AFN5=qzlF>$@v|oc72mXD|NWJ%Fi$JW{vf6-"
    ">yEd*XEt^`4W$5jw4_ZjtF4pK#M41viL2+;{cdpjSTXF0qfte?PD}X#MJ#Y<aBXD$0LBHK!GVzkw7s*ps_X8x3XF^V>j;?`JIyNP"
    "5<D5@FapClk>TNMv_M6Wa2n3?&=$(Fy44K|w7PJaToxs{GWbhevN4Vo>j!rAOmUyj&ImC9MZf&xG8^(8+*P(+sv3gEXrAe;s$LLA"
    "r?>g@7&|2DNGV=T5Oc5B8k)psR@1$9BCMb5T6k&}L32x8M&^<EACCd>lrH$q9<?tYlU@2#I+T(SmfWIHdDkfo(d+oM@NwZ~8<(;*"
    "TeX*D{>b71@0gP<J<iPN>DNDas$*%K_*XvpvFyX%sK$!EPub+ww%9qYb(l>y;aIi{Q-&RY_gLk^qf{CkATV;F3q46tani3=Yyd-C"
    "KLDe290TYaCcks5H%U746ExGJdqis6kr*m@N(KjAxu4MXtAYMLo>DxzHSv%j%z7P!B_+u2zDlvNg@QaI2g4^ISKcK*$VQRhrhsNv"
    "?1C8@Hq8OS`Gt55m>}g4rB>HTlG&my5Lv*JxZma>YH+Zc|^lg=b8X!&zU-4Hf?fF|lmFT#-"
    "d&pE{Pe9h*Oc>=s62{!o!ybvu@9|pM9Ue`$fJic#3=y}`Ul#mla?93UcB@`^U$O$P*;-"
    "J1Ihu?I3dsRk&tSDAY^XPz#}?=Q>t|oOFx9YAiK>u>Aqc$|Tx$|`_nNG&8TZ;`?h_BdN9EHdC7cXl%)Zd4@8lR&m$&9FkPM~-"
    "o#+>R6(!g2i!cjlh`qPnuGA#!xaKXC|955~b=AN&Ag~;tTW;_k7VDHlW|ra5Hbhq9n767!eg`93CR%+jxKFptOMn<_DAI#I79s^="
    "&P6kipPul}?%7+gO`64<YZxr`>2z^hC(;;}5Yw&Rvf8yU?>WZ?X8ijZ-"
    "+aIUoy;idDei(tp=q=xVoV+JRNtK&6;ZMYp3%T4h{Bstz2Ah7>os2Q+z5mW3BD@p3>U35l_d>!N1@uf;m5dlmwt}8&ab2xYL-"
    "h}&suATtmfO-Axm8Wh1eTpR{H(R*5k=%Z`q1tDM{r9gUj4+!liw2W;CNIB)Tb4zj(yqB&jXQb^YKyTju)J+Qs<2^fXtnSzLZb>y7"
    "p_c9Da57h3hp=Z-w+eNqR%9bLx?{{-"
    "2h&1!>w_<jC9f5bfrHrt^fbgPUF>4~O?N%!zOH2Q$48GOJB@tr}R3pg_snM_@1)6A~6?{*FS5QYF;5WwyE{GLh7rYq(Mo1Gm7mU>"
    "m^HY@7@?W>>{23H>uMY93p$yB+t=u8`>M?UMWYRyhkVHH`(tj(IK)yi&Jb##2UVtCLJWA~+_a_}VV@t%YuidfdBsG@q><z&z$kS+"
    "|MoGJYU$S2p!UmV!}%{KtfnGNlhsj#>#ze-"
    "!{Oh1_fcIM+5F?uLPBFcvu{fV}XG=!e44jexg0@klYQNylnX7#td5l8l*^_Fw9!pl+G)%bL$Q;=44gs$4PY;f_;i|rUzrI{}-"
    "w6|{SaV1wRMn6T$uc+)YF={wM(MA!}QqCNNcoeMg`gm%NDs(zu_tU`qjR_*)#gK=)_O(=0SuC-jbi@Im<v|;+B)#`-"
    "0&o(p5qL16%lJw(vyot<;Z0e}`){#)bh<-r!<>?|Kk%#j;LVn-"
    "7VjK#Er7LczAgVOSzCR5kh*Rme=F_=FG+$=`?u*Wc{+M&a$X63ihp+&^RV5k{hQIq@ewOxek2~c_X(-"
    "q*vk=V4yj_#gkZ?|Iid@R%<4Nh#2<3lpUY(U9H9W}(ZQda^3T|A=Z&*o!o0XjgaL+<9%oBR)+a`@C(E2?RJooJ-"
    "U(aGZ|tQ3&OF~fVVy)Vi5|)POiFjl*FuAp5PnU-"
    "1qq;0({;piSf+bU_^cNaaK3fPab|M+lCXAAkr&tyfW9d>nGw33VGc&3U#LZoYlin8^SjrwD^{-"
    "8GG5W+!&7J;Sa>Voryl(kJA4ZPqM3vFBJF9H7E^$m11jKBtd#}r>8EGIr_3~}b(GruD{C%lzIOse!Z&6deYZ|tSJ|q%gkVfejTQ<"
    "SnzZ$plnAFjluP>I`-}Pr+kDd`JT-F2@>0$$jq`Rd!<9#F3(xo%h#jSme@Yp-ybaZO`6Er!?jHH)EU{uXi4)mcQiR_Vk}#QVS4Fo"
    "@jINNtp0;EH=a=9nA0Xb6_-"
    "8Q!Nn@Istz#86ygV@3keEmRhO{C!a~I>sS{GqH>R;Iylk?NWVi#G_BgCQBkTDY}&l$MTIV^lrfGjJ~+-"
    "m(!gqK57i{Co7GPt4*Wd>h6ge(~-"
    "Yf~vt##&ovwC43Bmq5zf7MAGM<56=9>n@jB6{hDE(z9))Y@545`W8ttr97bRq%gU1{V}0bWQmP!W=a|Cc5A>P%cFGpc#JD*AmNzi"
    "U(CFrDq^N}t`s|<rnFiszR`Vc6|%C#c&hyI!!%`xnoh1ap!3@QT*5KTUw*%G2Nt~%uAd2O8W^gx7m1OkCwRm%gb_>X@0Wo`U7_`d"
    "y%6H#%$PK74mD!JDh(+61W;I`T<zf1fd{M}s8(%_w|r3Vp7@4s($3hL--I|cnIu!GfJ}_321mots03M-"
    "hPA5m`IF2zXldI4=*@Jmb{bFV^7DtDv_Mj%-Z4Vk7bGNjUB3lRs$Deb;4TJMXL_k{5@<K|DpL*%_8FU8kCFrj{(d_PuC5wj-"
    "LX}vwXs(lC#5D4d8P=R%*NHHs|XR!_}+7wnlUDjdBdD&FDbFgS>e@N@8R?gD-YHaJox1P3Z4+$vu`7)<RfNVu+iMUJP5&7Zb2@9V"
    "`98L0V2;R9fwS3Q06LnoPh(bNh$#A#WB|Y-<1U?@n*FlR&}&vXT?sRd|_X@-fhRU#-VyN2G#=ojE?0FOSvYe$x-"
    "zTy<a#tC~|V;6xL@w9&QcWJzGh{%Q|ItNaaYmV@GJ9$eNfp%k`<OG%bF~^Itk+>O^S^whD@ct~5XzXbmZS0;pz_CQ9@!WYya0YF7"
    "un3GvQ7Qb}%W_Te`xiF-`WsGtUq`qyoI0pup;hV~Kcyft=`GKZp|T$s+(ib0K9a^bsE-"
    "l@~naM#@`>B!AJAH*!g;A1!BLNj`Az=^ZrjbLU=!e&Xa?dh`9N`8E6F;_)_d<{Q-KmR&p(-3Xlkh-"
    "=4ka`4u5A{7vW}}2S;Fpm&JKjpcW%sdkrC_ZcOnfxXHIldMeMyy;Ad`m%3n1t4fCPQ2g_Jd?9wDUu{XVVNvj#a5NEJF-JGp!-"
    "XmhNeO=b4KWe}3JW1U)vtfVVuIO_j?p>Oq+dgF*9q_9R*0?9;Bnr!nJ{oVl0)0_C~x)QlpU={F5ihaJ45ZMv22R8hCO;2&8_x>F="
    "6LS_y0)KX58M`0){eZ09XB3?EsZ$IjLG1TsT!SD*ZK&n0Im0%R#v09dt(C29IYqTni_yF`QgTyUId)MijC{u!qxzKTr+#?mFzWp2"
    "2wR3D_xmui(>HYM%VG_fnT!^KZyELflM0<+B*3t0Rfr-"
    "9{1P@atKT#C7!U2)AP(XcN#ub)TwN9g^XH<ttK(<|fPE|#Y*N%(aCKfT0<*G-"
    "sJfHm1QDTjtKkIU+38*i{i1zxuGQql`k*)f+9jy$v#s8O1*|pp6GVkgs5_z`%=bvb)J0Z4K#1U$-"
    "e95a`r9Y}d!%I~tT8kEMa>>(Y}V%bS7#&QWP%O1*tYhspm9Fs<(xQz7&G?@%>!cWPVhNv^iP6b_F*!F#C+~#|69MM3+y@+7jf*Qa"
    "Ne<48C`8-"
    "#uN3#B!?l=(~=X^7u7cjh0CbF*fK%nPY$j6n}ff>dtmOqJ8*&i<bnca_?`4LCB(CY)Pt!vo`EQI?u3gD;vxxSM&%mbecm+7S@E&T"
    "4zlU4l=!I4CH*V=)u77|rsN|U8ujjYc@jvMS&!3vC~~?(rtGE1Zo)?ORPK;yq^91_+$a&jcZ<3nx|pa*u3(J&GEaw0Kx`O$PEuB;"
    "2zo9JQ=@1s`*+R?2Ttukulm}gt)Ejut&#5Vf*~V!B3!KfEKN$aiNw8<y3LD?&N#04P~Qa2gx4}bIS@FG)>|cewAISA5&Od;X5u*M"
    "zutrxtFNj+Ae4EH<yWbgwc;=qqOH|r8(sF+oEcW6ajx|4vcb{UfCN$mHH<zujoc#Z)1Gy*z|}KozFta)CFoNlL@UhaLj?H$p-"
    "A~mYH&<=<;lR)S_ZE8<}Hy7{V81YItGlI$hv-"
    "c4OGq76Q<W7oIT+p*+0fr{Z9ur3AqW`<ZW2HCYK+DpuWwGvD(CKX@hVz)QsPK|DSSkRfiKUuU;buu9PJN-"
    "i!!+e6JSv0F3Yp%Mh^Fh}xqs4fdsLmBW0zNJC3e8GfZkY_Vyrw_)>&$l5}(iM^oa5>KS&O10}xH_F{-"
    "A+Se^+u$~lP`TKD|1+hiF|g|NcVPq}T&2ayNAD2zi%wDi3&(0&pD}Ek+hu-bNz5?xbh+XizjJ9H<ChRVmbl6H>asp#q!<^^TA5F6"
    "oSD2|b8R9^y1@YU9ZJU_pKnssBp?|J)NS@_t%Po8#CY9J6XYl%UliSa_t@zyE0==y{_@72kenI346hq;5t4$vCIQmjLCh?`riPPx"
    "m#<>eVn$}PSLDc2SAx2Vt>YMRlyHxLWA?}@KUMpc1CIT98S18^NxNVY)26dcMG+64ZmU%_d0Ko!qD&@MQw-ii6!s9XbpxxW(Um%<"
    "_xji(akW#=uYNb-"
    "#XSDfkQ6w!XF~d=E9P{((5hOK>>WNi7hC_CA~oU@72$b}q~0l;vfvue_!&nMBnBMMXKR)KzX3a7&9zouMJE^@+3~%<sFsA(Sradv"
    "u&jn8L=H^&`j;v|dXsxf;!y`_8R)3Q;3taHnrDo#+b*PgDUPOiU@{MGYV0W@%*~$2j6PEz<+r#kMO5ts5*@%WbEf?4Suj#W94_lJ"
    "NKsI*r)oqxr*4H3AG9Pg`k6t~TiL}R%ThDOMuw4P`A@CdBL`hHy^Di&2JDc6vG)py8;0nc?aP}yS@xbuok@@hQpfYrZ3BS|*$aT{"
    "Gm_ttO}xRV4hw--*3`{-"
    "<9F_yEuVB_6UHu?8}<<q=9wO|z|1|4;6CIaS(7hYM{{a=SB+<;n{74HyCk5#F0tB7S%W$vg{GXnGd*Z!xbOkRz9&1<8TD@V^EEs|"
    "7>9Ahz8`Pto#n@~M=QBDHf%FxG9nN+x{SrlP)3mRn!)_mPn?ndntYD+>BaDiHl_LYtFI#n#VJnNZLN6sbEC%LG)p1Zb*E8=uObS`"
    "detxs%@bnSF8^QV8>^95^5XyGoBZ4qfkW0sf4GuNeka^z-"
    "d*P_nNnP|bKcHQ>)7Vh+0(B^n|h2njW9@D*YAmY(j|@IjRy^I%Od*ySmDlV2?v2zH{aw8!y2x|wS7*>9ec(dgyZDYr>i#ZvBjRzk"
    "{CHxE<uMOZxt-oK7zTJR6YilF3CVbpg_49Jn-<b8y&yUu-X(*HDo}6eDV}c{dex&dn7MJG<~tWvC3)#5-"
    "b_WGvr)SDexobpTvvW5P1a0bXWgG-"
    "%5G0Ekk#rKcqTew{HFwkuFkZb|H_0DEF2n4!T@ilR8C1alo_u{`=AwhKu_9x?*Bl*UuP*AQUKn$DxI)63!@fdA(*vc5yv$AGdaS<"
    "?!Z!J568LYKxh3w?sfEigW1QWj8*)_~$$tC*U<4>C!Q-"
    "HWCY%xOBLO?`aZ!vwJgi31PP}0@<UIWTvt0X6z1fS@u$MzG>6FloN+KL@u*ZLRdoL9>#f?4qdD+rNl_uD>QEl=(khct-jA-"
    "`n+uuzVaA&hQ+f+^*!h=DHVIg@JxJP#DXk3XEc!YNHcinjC&>uLnC;rSbhyvuJgX$GZrQdcoNgsYV|9_RrtIXp1J|fFvmTG^Xf8n"
    "UO66gqSd=A&|JArvvocn8{bV#V0y7!oaYt+4`NX!56uwXf<h<sc$)j*uh5ZfN01=Z_k%++6tc{n-"
    "<;Kfrq1zTo*yd#|Bv%hwsL=zT)%eFXO>zsTqkQaIlA)Qol{#7;b4~R^X-"
    "&0CgucFx_j6HYreIFGc3)Bs8f|$+7@)+GYQ}#)+cps45eXx2#?!V2X6zWF>e%UoVIoID{=3N2y~R3ZMYVv#2W~fd$q-"
    "wzTc3<M6QD`9!nI`E-U%B&Y>Wt4g#9%k6x>Ir{p+c00!zsB0&+F3Gh3Su1}KF&!~nwHA_pY?u^a1;;bAhYpitXv%b~0+-"
    "xSowYfy*k+(<KkNlP#YQ3VFGo4)GN^I-"
    "n?=uy!3s6^~VwrgcI<dr8^qD%ymhG3Y)n*c$z`|M+w@?z%{Uut`WS>tn9`QR>0ttoMr>^1Ub>v&}S2B|{6t&;)+m;dDA!nQ(p#z6"
    "B+TA(&hwTQg*Z>2P;kM=&xn|(NkADZ$W$m_h(`BY(kQOZ0eUIFHN#H@^Y8q)kj>xt3k{#7|y@VTngzG*PH02*C`kFY{o}?LfEeZO"
    "yzS^3@e<Vr8BHV0ancS^6Q&!hOFzTt$PVcCx*~gpE@n=#Y>$yH4v(VGBhNSHJa$sIv9ix>DkSb^3O^g&*1kt@+lwMZT&$=!rBi(7"
    "?Oyz`kAoMWFjuTjb3T(<VvwKo8#d~X6n8pd)eai2IU6_AlM{ngbi3Ax#UtzQLMAAVEMH(e?$NR7h5R14lMacAGJ*>SXynUrqPE&I"
    "U`oum%j)WR+@~O*zUl|3Q|94dV&)Xs@pQ+Z@KRl}aTm)CB6YDkp=lEytT5ed#&44uk8Xv~fY~bROC_-d-"
    "p7)eY;hT_1t&N8^A7%Jeuzaq;RnzQ%cFr12*JEnk5Q+TROxM5*NhUVFseTV-"
    "F&Q2}g04L};U=@=tYy*km<0A9AkfXdWTxHD<rWpteAWI`5m*u{C%3@E2xpPtwHBViKJwyvp^|M?Poz0eg3Qfm#(<XKR|2Um!j1{>"
    "!v8f&&m}r7!xr&w1-}jjcujY%WsK&qOWh+Cp)nnk*C3TOe^wp~fc-"
    "3ij5ouT!zJtqx`A)#I+n9dfHTJTWIQ>tmTwJdEy@L8RWYKJV@)~3wcI=#D;O`mb@|mBGI~O!%@Dk<uJQF65mM>3m$7<h2<!S~TAF"
    "z#%}oN9%Yiw$4~h5_FUTO&b@v)?y8C%Pc}sw$GJkeEjyV|F4*khiD($%;sL_VLMv{Ab!s|Y`tYYpwK0O1myno&$WAE#aq7+Ge7}w"
    "qQZ>r5KJy_klCk<>FFC!}78i(2<u7xLij)^2Cu?W@;>T8^6o;gd%<k+;sUS{SdMUf?7&R>Nhf<zH*N;FN+kYIGl9uwEN=@`_>+W?"
    "rjq#uD${$SXnehhwhnM-s}PT<w!S!Eygf4s5-"
    "9s6B%rNtdto#8qGQuWSrEaQ@0VOva?hzFkm({zrHtv<(_iN&#X0J~J#g12k1nk<U5hiFOgei~za?giJf78B=7fWW{qWNiXFdS=a9"
    "?NKt3gcU@&ZiyZ#UJQ_T&;WIg#d1wRu>=lOLV?t>=ar@Iq?Y`u0dAFu_RF0j>DLTSJP0UTf7xq9x+Z4$pA-DV=wR}ZIsZBv^QBAx"
    "X}I?3T8TQ<^lt86Gu57Ri}biOYc(wf=nY4Ee<qYPTYhx@ZFn+fV(L=9ZKd$^0<tsJ6i0kxtvyO&usDoke<z|75F{FufD5-K--"
    "@EyS;@Gt23ff$46QNh6V#XdJv6qOiLW*WCiW0}@1%Hny^qL3vgSFuauRl7Dv531-WK2veea#fus&FcC4a{-OHgKR#fGgEwS-DiZu"
    "K^qF3^5G@$qjL)CjkbTTxIqqb;a!lAN<L*oT@EbGupP)l%9-)KH3_NppMpS-0Lx@5olrUyx}KKmVD$#(5A-"
    "vKa6U`KLRV2}K(ZZaEQ<QDy5fvL0<wT59e+Ldkdd$AxSqPQf}E#n=<;H?d?i)-"
    "8zipTxiXd_U0@Ka&SIX8W2iy*X*gzV+$vKlgvdR7XRRVa)Tt|8waf0=){^rZw|3X#%X^hYS)u$=aAFSE;38Y)Qv=Xaxa=kbacKfm"
    "!PJ=Sf{mqGtO99lt<=_t{z4PB4{r(xtWzo><~FWj$3t+%ssu^R3oIWe^9{XQuR!YOR#x8eHR8Kc&m`9hc4f8e{1hrh7;8YLqUMx^"
    "n`FowqW5%$Lsf9F<=ry`D;j|8cg3h{;-0;OTB&0SO5g4dLc=O^X_Y!QGGRl2B={^kIiH-z2J>k-DA8a($E6-"
    "y4!Z>Nh*^@1r9<q5riQsGc?hd6rDQPQNSctYqvEB3rKsjDc72Q?^+3_R%&<QE(#-"
    "#N~Cqd7jr$*9y39lR$J@*)(kjbP9&fe1FCutF$npwX0{#pblacRT`T?JuFz7SL`vyG}jC@!ZdNCEv<p1fz4>#Xm)nXzXE{p{|tG!"
    "%am*du<$S3lU5^W)*`>fPiaEOLkggY+l>0A-"
    "&%u}BZKCH?vI;|=Xd6Co#}t~@z|Zo_!0ZvM?GG#pm0oTL$8*O?nMgUf=1bAy6MR<ecLFny0+F)gqXOF;#Kb$fh+~j)s{ta(mPBX-"
    "^9GO+1^o!hP7Asu!LSlRdTv6n2j2f$nfxb76L_82=cUQ0b7s-dSDW##VAFXA&lMmlHJq3O*H6px@z}Cr<cw@<V{Irm-_0M-"
    "Q}x*G<rEE4ttAE&inpi=``gc2ITUr?h_X3(8g@iB*TSPXT_Y7B)$77##d(GtDsdK<h6ZXswGo}E7@9;bYM;6x9)viGt+s<j24x(E"
    "$a2E5&7LI7RFxUA7m?B3Q*l^O5NKTUuIocd--IqSdnT@yV%#NxM&cMujqG4ABmbL;%=l*GG{hD#?b_Z8uWku(lOgUPr!Vam$YsB5"
    "zU&krC9u@GYvD&CE>J?>Mcc9)a-XEv2G{zyD4Sy-sk@Gx1Pa&NTJyV!Ry~tnO|$O$d6pq$vl0>4kX%M!D9dCQD|OF-"
    "3hF7cR&R79$sPb&;rRhu2-pu3INf{o{o_~%Q+%zum*gGSNWt+eKH4cL{Y`~?uq}UMj1Vc^v%$l8H={>AixBOsU$X@nFVn!-"
    "I;$KJ+wsj<51;YF+BRGzDsX}J=&FY*_t?2GA`Tr*jv+oMzL+<T~B`;BNp#qG_7_}3@7-Sb)-"
    "g<k<2bpq=w|)PDH|Ut)#rq5nwp}7UW|+LY!2UBw<Z)l!{{FC8Tu1iEjWfh1*W%D?YN<Wc_o(nu#idY6MsBxR4E|fK;U&Y39Yh6L4"
    "h1-RD{sp|VCpydBr-8p*Nop$X3zRsc-"
    "=J^q`TL&SxVqf(4~m#ekis_dV29=a>VL{MZvHOKJq!ppV_D*@n|Sn{JHsLjF3G<e66noNeyaHf9_8pCft<AUN4Ka}CYnY4yZ2QQG"
    "y+NTPkEyf?W#tT;8I?PP8Rw&0W(cO1KE1l(Jqc%xO5B;TmXL^$AWU0Rsu||7d*yz{?_lU11lY?vsn~A#F?>W>vp*;YEd;3W>ju()"
    "l`}OMD>dDsR?GV;pCVCi-^3F~YNix35D-oxte;F(kwj$HkwOCE7x5e3Mih7LdoizlS!iD-"
    "2?Rv==5;4zs>383?4>r6;PK2vgO)gk^15{=7oodMi(y5Dul^^5&rl4u*@jeSg17g8E#)EAPd4=F559651m;mxlqKS&3XLd(Y4}}c"
    "a=AIMppc{zrQ<*v{wS?v5OLf%`xWraD9>=6fPSC6IfLK(pvC;UgHfsLD?eFpN3M31Z_TgAQ10!W4GbHzPDRIr0cxsuDpL2XaQTi}"
    "bm|b@<YM8o4a_4ZWA$sL5Gpih`v4I!bwOk3>kkDMaPAGlPqN-"
    "(hNg2Ixm_Na>46_mD;7L@d_==`&JbF>uq2se8AQvZ5#%6YVXD%Bmy9K-"
    "1#{~2l*}eXoi&}gv#nkrZVcwUw<~YH#aNh@U)JHt8lD9ah|EH5sJ%!2DkhR{by){dLHxR}97_$Hj{ZT0Mt)KPv&6v*kn=GS?O2i3"
    "iESFYqZ6|+H1;ijO*)-(M+Y)+BrnQ!QdMzAZSUc}Kt-@M%D*-"
    "+>*XM`SWDPhH33?5~>3l2DZV%abs9RSafX)1S4*AS{fS?t0=P@8WRpOp@aMLAUT?XV_Ok~A6j&m&=xi|e`>-kJm-"
    "#>X+`3g)hY06!285Op_`-"
    ")rP2;|aMp}JOdTBBT3%}JxDqS+Ir5fobAD3tI^^bb^beO#Mg1R3+1+;?yr6Ss<TNnZ3bdWEb@(4qL}{yH^dFVQ-_hxWkg`>uFLp5"
    "9P7?X~AzVdp0#Ff+Nr*TZFOBBJ$9P%XE;e_b`Di?P(!vyD~N`zpiZjCXAq5)Ki!p6_GKJv6ownC*#L_G3Y5R$80I{yMLm6`p5qy{"
    "&|*R_^{qwco&XV)!IN%8;d_m9}v8RN=(L-"
    "4&Yn7`)I<Mz&T1Y`01{_rXc^NyUZcwsYFG@cvB|>{Li7L{)DbMDY46j!qxGYeEIqq#pDdH^``HKa%NO+l}l%PQAhTMMq@G{J2khN"
    "7{>-"
    "B${#~L`_<R842LGt&7p{OtR2;1zah>MqJ&;poJ_9Vjl~EiBiA{bsY2op;k~VnhFBd9)9W`B{rj|T(!Q$(Mt@+*+XDZD&;{wk*j@+"
    "$%8%RojCdH-d2W!fC$c6&&j*<=!ZcCKUA;O;cnrO+LC7wyW|o+7(LlK4yQc}Xol6)lOu<+HBZ_vjaG*q-!_q$^-"
    "Os^qmUj}iZSn1H$#LY*v!l&HBRIzy|N~j)i^L0rhk?Iik_`1QxHYktRRYuN@dTvq^wTJ(LUZPvhX8Wo>4XDlI1@!%{ZN!qo5-"
    "<=Y*#y>9IMT;}ufiI!S~OcE#ECrF8w9w%Xwr^3~G4kmp|~C1$HNIquN|qcMW%5k;g@d`k$wj8}lZDqEBM*Utp^z;IkCcLYI}E{9)"
    "{?QV+(<DKO4)brZ0y|aL5Bz~`FDg-6`j4UCDVn0b*L^DsdK=g0VznTPUugimOC$x0UVcq&fWE-Gf6^<bF`d6*h*?+qkTXyMtVy!#"
    "Q*kk-"
    ";tPG*spPb7@WC80&;~MS|Ud|_4xwTCaN7T<ICC||vwsC=jA;&`>7m<i0tv*q|jWjsIyjs(TJxim3h23!_8i7rGBXwN3u|bGei5Zs"
    "VNj82r0cSgMXkM|@?A-"
    "%pA#C1bZ2oQopfuV%{)cEb0B=GaM3orQw*M0L4USk*+RK^oZKM6#gzTEz?xqn0`sqZraD2Wx04Bq}io>{}Y<|SxfG-"
    "}W=03RT;z{_(=W<sL$bb9lff?B<DRYepm;NKsZJ7?0h0n0UcLZbD!(tf>J3j+M$GrdjOClt}r>edBlwO53F&&=Pvz_yM=-"
    "r6d%HLjRkcD(u=)X_7^fSU<dD)T_(rTjp&9xuZ-PUVViw6r|Gqh`DN4s$hzil`oTkmqV80nWLW_LFHl@Z~cSz`ZlCTsG@Tr#CTHt"
    "NK8x@jP4B%nCDNZw#vOh0NN%yAMmn7s4q=RBP;UO(A$H)tvR^)u?XCM$bt*NCv}2+pe9Weh|@btW8O*ew1Dsjn;ML=E1yBHO<@>9"
    "e)R*hT<2WTI=MLClmC!l36vYPVLQx-jFI&r75bq2Zg0yXA~qxfTsDo&5h&k@Ac3+xULEe59|vq84}>GNu&Vk~>z}K+U6+foAf%!^"
    "3th9_xmK|M!sZjZkuHuYhE4QiaE*CBv=)z!+h!9%6^3z^J#ksWh+oRf|ynx2B>AzffK?A<c;1W{(NYKRk2)T*bfkhFz}9EXoRuIO"
    "d~o0ptBkNTR24{%V75Vt)@ods6WXe&?IZoQtxC`5_%Axyg2R9~>0Ze@w1N7g;$6e%(*z=M!-"
    "YN9X1a5n~z3oTz#Owl#rH%}SKjok(3uUhTvOuK#^rU|rZqy)cQ^<xKj|F<hgi_H}f``}S5?g*PU1$GFahS79Tl6}2=urp0BVOBOH"
    "P^gF3#kz;AIUedQt9w8jcfHKycDZ_s+q7lhCRQDQYZ=rjWQNch=U^1yt_^;+$ttJ75fG@UJ+IruA<M3&Esb0Q#l|4+qt76oeD;<y"
    "~ai@6JHm4akMgRKyvXAZ)b;ACmK+&Hj&C^qg$6%zVudO}imE^s^3hs}#R8$w+!1IslPV+!ZZxsGXZWIz}6m1#b>Ci2ylC#SFGD@1"
    "?MW$jp{2&AnGakXYG}TdeZ(p8(Q;D~t?zO%T_c(lnaz2wBz)i#TV+PY;r#p)%Ds-"
    "jJ{;fxobY(Su46xmK;=Q5&I49q23xC!7>UX~z(0e@Lt5WQ1imSvk>_QRKK#qFuiiYd-KzC*8Ikttd^Vk9y=>#cOGW#U|c}pk-"
    "=Zb4ygP*7lAA9aKFL(AlU5|S=Ed+R7CTjPgS%A*kVgs8&C@JZheeY<osm&w9HH>TK?waH~g98fCrbkd?cner+R0e$(Tap{NI!i`z"
    "me5M@H59pyN|%n5v^1GM?U|G03DCI4yZUt(N?n0)h;kiRZm@?N)BO7S6UoRNl2;ng@yXcm5QFotT+`bBiU&WZa{VqGT~GxPgP}yV"
    "cRo-=Mv6$HrwxwxPx6&^t-"
    "U)+;yVa<4fonJ2nsm&e&8Mr%`oFQ%?ZxsY5jzn1>m0(X=%<#VtcZ=RM6C*89<5e+}vqvoiL%N=;<R6?WfzA#DA>O1zocnpil^bl!"
    "C0UxggSvC8_3$vg0dZ4*Gr7@6&J@dolJ7z}_;^tU)7Nr}6dm`W43%EQMt2?pq<+9o?vv7ob*`aK6|xb==cn^QfJOdVg=Bv+=H`QJ"
    "4wZXm(@0!MC>kec-v-_GSlLVuk0jMW<d{lPxTuBk14n%{(jvG@Z-#8Hx;pgXKNZVe0|kjBI&ie=W~@wY8FA!laGHe-uyAeFm#YG="
    "Dc|l1FDr{>3})=`-"
    "^W2~DAe4%~jz$Hf?IT3EoFD=f9dzK$1*4*!Tn_E!(LAN43104dKo5d0GoCvS)+`kmugA9N|j+@x(5IEw`kAYg~6*hw+f8Pi?Er7("
    "KOj`r(p_i$LRurLiv{(71SFWDzp`)k%Jb^c8n-"
    "a~2FxCb*NIv+msn)C9MJW1$C=t2idZ1Mga5K&W=LbdTci0p6`9j(57v`1%5tM4dmkl+Nf{fm;T#bhR9rMjzOG>LJQ$A9~)5UX~`F"
    "`=5q1RDgynu`r|lZJ6SXE@<ehLPCb7N6+LZ0PQIp30%~lqrO1?e`pwYBH%Y_D?SC{t_*nmqa6q=IM*UI@Vs359@d74C^feqbsXyo"
    "#I@cYQZF<*7|9wxXrK&)^CRpyY#Eh;JRDC)SCby)NWdy83|Vp$jAPe?VUrb#TT-IyQgk>$`X}n|J!~O`xVM_`lQacwq{DmOCFonV"
    "IbNWaj!n<E(PGXsfVyEyZ_<Qx_z~J<*iHXC(h1`kOFC?rP}0o<}&L<Ic6UVVre@#zuXcPy%ay1&FX#Ea;<wr6<A%rEtwex-ygGCn"
    "}_2ejhJ(rjFemVa3zSI>SX@IL~T8(Dto|*SOL_<&9$`<7+MGc?<D+Mm(h16L@(Onwc+1;gwhYP=~9b9ER%d!GT+v95W<l-"
    "kvUo@bTw)DhfWbd{MPTWcp}Rk)ApnDz?q^MI_wdL9cYs7sxiLS^7IBNfsg1V!sYtxyRy|5(Z=K5J4Enl(XIqm<G425pn~C16%{yC"
    "5yP@*9FdO7-S&z^d*h?qr^5W5*L$EQNoJohY-T4Z&gr9%&%S6U#b#kN({GMpx?(*=#$4N0azc__5Ko66-t~_-"
    "OLn^Zu~>vKVQ{f!x7EVxEu!Q%;Mlz*E7MtE4_Y@)yn-PL$=Bml*xKBR^6b10y!!YgEE4lLK}>wdXhvaZAaEy$yeCzrhzIZ7mdaoz"
    "pxN&nq4tlpeYeYTJl_#o*7WO2FsBCV&A(m&;%@mn_rzHg?+QmZ7#05bw#Wm2mziApOMEW1CC#eKrvjDU2(B$iXbu4>8+ElaJkV9s"
    "ZQkCf)(WoScsd8q>x1SfK>{Ub_i2z7EjwEW<ixaKG!Y^N`~>cXzBtu6#zo|*-"
    "wUE}g}Yr7ir#)*C`J|Wx=dL@$YS`<O%uLEx723}Aiw93EcZ2D<Rx0bt<F*Ih`8j^m5vKipJmJbM9q49+3DPDCNP{~1$`jZeq>8+W"
    "{E96a%?1d7Fu-e>4Kiv_HYd?uW`5b=rWj94h_iReB>a=sJKTHW=qAxxt6tocvlees2tYiTT>~}7d<ab2EBVDZZDOv{M`Gq@B-"
    "R7WZyB&ud#}-$~_+37Urle9^Gidj-"
    "N@!`aPAg!DjYY<@B4dUe)>WtV3xviMqnNSmr)EA^KFCXv}XXcTjg%2%(Kl`w%IS7eqo9!=C6~WtXbEum8X`*?U<SEv+rs*#mh6N%"
    "qM6+e7bji1#h@uY5BeKBz8*Ndp75UP{5J_P6<W4Y2~@$^5z%Qd^Qq%qsVBxZomSl{@)95BZt&5=0EXo(8uB7KIzZPuF$Z%JQ5Y^A"
    "%_3G>x$Mzn_o<Z64_GAA>lRZq`x;>#_`zwl%cTp3AMp5=bdjluLs$b@%F&s_Rr6;Fps}<Q;8p0WsbihUM|L#b1IHY%1yS-V7>R6-"
    "fjs|DU4~z$a3mXP7Tx{otb(cvpxD?~D{}qLOFFfjaU&(+LT2dbtiaKxs~Pg}$fBrO6=Kl5rtfs9Bmzf=s~?$jxRTd?%Sq%<a*>TD"
    "qboK}x=n{tcsYcuy3Ov(t_#kn<@lIa%9~PUW#C91Q9AngB%84Qn#uz;`YLTR8h*`mou<FPV2zYUGw^v6gL$uthV(euVsXR?)P&E%"
    "I3N`vjJ#*6s#cIAZt<=zn{?5vD>%9^TN8hJ(K*(Kus^{)jTJyt4w}iKtCXka5hHZqp$XK2CgEGiU8yry3PSfMxt0$2FQ*-"
    "3XNi@n$2yW0Wz^F^X@+A$5hNj^_I=<u7GbL}PoPDlNv?lJu%v_!wOe+k;Vv#G3ic$iS2g`s&}e-"
    "AOX>YliD%nKFQO*f&~k^pbt~fN)Pl+O$jC)^R&8j7H#SzsBWio@lT^@FRO&z*)j+jIt!F;UIyb6C)1F(Y3B+{Kg9OGZZw7=*E(hZ"
    "S(4SLL;o4r;Aae?UT2VqSoKskbgRw6SRQ!Z0Tj^ng;Rrfq_4q%LhM%ckypS^Ne)?s4@eDJ0VZ+fRGZ*^oBlBV$LXt-"
    "=om3E!ese+YDt9RzmwMujIPg8-"
    "5rVUjfDo;f)>N%n#$Sq+(&P@4CZu>xyrlwzajB)c?S$xvG2P!|fqPL0fn<Eb!Hn1?w6HXFnQtVGqkOykJYAB)UrZ&PL{%{O>2Du|"
    "x@6-AW%pbOvkXuj}lSEy)y^5I%$IWr7xPtMkatt-434YfVa&yN;fmq+En8#n;(GMBjBJ6grqzVRNw1Z!6gLX+B-1*6-<-k8*rOh-"
    "V%dmsQ6&{T`Mh(MCz%7$=nJp!Paq+jq9AyE)up`8IRXv!Wq_Pxty(FV0oPYWP>~>F()85zemxq@?6ArT$%7)Q){*DD&SelBS`cms"
    "~@z^JspyUfBTbi_W)our%b&W`JVi29E6*u{%;<O_oP$)C5kUk&K3wb)jMc<1n~;dXmlCku3KK^Du#S4p<GG6FS2_e2}jS``TJ9Kz"
    "E0(Y=b+27y+>-u_T=Bqo<<3=l=FxGBR$>R|+_UVNhn=WE2q;gvt}saA&ans%g-"
    "a0)2g7#l9xXjvA}?ERo1xNn(Guk?3ohWe^bn?8^sq!*&z$12fR3G{X8vHcMh`zw8TRKV3B#-"
    "4M&}@?4oTZ_C6}nii0Qf+Ds+T1Q8Y8fA7T6m0@3e&3iDeK{tk_6f<zrMdQRlV{90ia4h*)Vb0k^9^YGZlF%$5yhI#ux#<$1AH)S<"
    "I}a-T#@M}G#VSuylutrx8B1@y?Z$DnsiiF@!T9hTu@-v3>{>2Rf}B!?LKi-n5HKjMvw7jO+`!}J()7H8jX4pb{YZkZr-"
    "E+K7!%&Pg-"
    "=25kwV_n<c?~OqvDN@apM$hU0rkAShs+<?_%~Zu!}Jlu$iOpeWv5_i#@J*fWU265<FY)17OVcq_;l2B){O%FV!WxqqEf(!a*2&sR"
    "+o<UK1hxz)F>B3Vw|Pd-"
    "T?Nx~!N^^5|}M8MkIm75gQY4!8Hq#OHez!K*w1LUrJ;eL0H;y*9NN9&3qTTm4vA=A=Pv8vCo@o5u#41$*G)U-"
    "gjy3pYbgof0%FMZtV%kG4PYLoX#I(EaW1N`k1-)y-"
    "U)4Qe7+THAx4P!n_jNy(rwOmcxytUl9^nTWfq_0|(y%&l$dD!QBxNyDLUW0LBJGl2x)$ZBf6A5$ADNIS>;j_s;24NG2QjBY#hD%^"
    "`+E!cfd61GFu%^SF-~l5!b1NwisSZjq8_cHAgz}#iwl0x#z6Z5>TP5G@uhWcB()L@Bgp5QdNgca8rV+L-"
    "JD%rm%>!EBEs`x|#$9HBBhzzd(_YbY<@wWzJr+I`3_S!0^(uDHJFEUi9-"
    "C9@T?m}tk2B!oBOp7t6FQwI`kz$i+c&3*$XWpCEr9(UN+hwqfvFed>I`$QnJ<(bxk6;=Q%M@HJDGo9nhN)*cjgTMzAH2C6Y_Kej-"
    "|se%@CA@kWYI@6#L7cXai*l_Q2C!bthm$_kT>jh(4XqBRiXFbk=YVVa@N1&c?PwN_jA<x^Bj3kkpOSnfyh%%PL7=5>83iL9enx3*"
    "y9C`yB+bNyd))H%TlNqDV-"
    "r0+6+F|1)yZHz6E^yU$p{?cw0j0E5>_bixuzH_B4nI>s|0nwq|=C2U6kziWO>$4WmS&M1sM>x+HN`EN81ewXu_XhYhgRd6>!QP28"
    "a<wQ&h2^-"
    "IP52CWsW3ouyt*YLl(Mz1OGI}_5!kR6&mrt(3hq3MPYy?|6WtfrhMr?4`Yd{%)WOcz4`QDY|@cXp(yqPA$|I)lx=B0gOR**}wg9u"
    "afiA_RI`mK35@@uJ`Sj*gpbPG#+w@zQJv-"
    "Y7%?E(39=XhPD1rOYKO^zz~JV<k^^muOWnC`DGw3f`b?#K;~rKPqO91nQ2Nx;*4;nPKFmaX|chGd63xF;hIsCO93cfv@iB~r6@oc"
    "fuptjJC8@wjeLi}|&zA=>unv^TTcKPLtOpUszz$ud)X-*ucb;nH>nlDUsbyw@`l-"
    "kR1aCXJ#Di)!+?v9w%CM0)O`kT``dC8|PcxREg#2NQdE!+OGb#D@8Lz1Ro_x|&rAlKroO=$isz((#jnY3&yx`f6@>=z@+|Sj)wl?"
    "u{(3*B5Jf&UyIn)(5N!e7B%9A^g1?t)oOLGr>boeD|*!-Ki(c(}c|oK2$M~fMo-"
    "{W0F(kcN(7V*F5_OaPex9vaaK`c>{zC3w`TB?iLndX3M5QGcr~VARZH@J@@z2@VqfmB}%-"
    "h0CsM{(`vHoERd#<Ftxq?H%knODi~C!NdO3e!R`mHaUa&`nf3G$N#lD@VaF%OVFvy;sC8eiuHdY_Zr>?qC=9VgL3hur-"
    "COE3N;-8%@?~j+F0~E$K)qqgmMP-elvQ2;N)&nV72BjTF>5&ya|Jk5U3RCr108bgcDcl<Pg^L=t)DoJozxZb!e>&jJm~3Q4TJW^z"
    "sLznS_Q%K+9o3#2H2`6UYNO2g3i?{?jdw_cIB(L&ZE6exR8Li?XFL~6Fth(km%z8x<+=c5<Nn6#WrZpoHKL}_PUiAfOGiE`(B{#c"
    "A>J7`uiz|X+T=(`uir1m}Z~j`_!FRTLeR1mX1xf*NY|w@ow<H35&KJeS=A83fG#k>Z41$XYLail^&KxBZHnw9=BXirq}zgmM#tU6"
    "|#lY^x+|sZ4o>g(2wQgF7e)(9bxZsLRNNj5%@P7iDl3r)3)42G5S%bmyC^=S@xC)`I_U_JwyEKr8-"
    "sAo~Xp7X1mFFpejqa$35mJ6xiRJXkb8h)N*kWkvpW^yRNz=Yo#S&X^zSw&X|NbOeF&KH3V&(ABiGwEuZ0=MiyTMR&qGJS$bYs8{r"
    "FmDW%$3sNoGGrfVbDrJe8B89htU&^C`qmjWl7e4ZDbcUf<YIA0{1XnDI%6{2+<0c@?aERV3_JhIpb&a%2n(sO30BUZx}xH*@#JVq"
    "OIp#6tl$p=t7TbY;S;K<@F^whOup7adXo>@|Mq6l6L%lPwkt`-"
    "qOvk5H?BK{fpH6JwfXK6o{<O&^JVit(UjB*k{6_xmL4)v038lW=D?5&2dM|8g$TC9-vwYz{0qQYnu-"
    "?3zAw5$gB&t`tS#~T1Ur3sm>4>b|oq9t@b2jv>WK%_$x)sF7^Y(bDUacHfbkLBteP(ugjY+>&a7OzXQ7AajQ1Oy_!of^JS`EkL9S"
    "#maRR`kG{S!>XNt%I>`Oh}=^x4OP9%Q+l+I^(|J;|2d89;5`jT`^ug{)j&CC}*K-(vS7L)VD(EklpTFQzzi7ckK-Ivn$pk`9ZcH%"
    "W(Y#bOL-"
    "3o!*_on*i@8S~LvEYgjETX5hW=`=zJ)h~5+!B<quBU|r=`A}ZRuzqEZfg|TkCCh_w36nlvuqTJt4aE+wIxRd4O_X?bEPbspFFr?("
    "-tb<)pD#2b73HOd$@?k6m{^5ccy{H?GPeegE1h7rnPvn>su^y-"
    "Fnl)i`MnO*2?y;^g(_C%GsK(`U9i<~PE^5Vx^Uz+#oLjF|Gp0a_QFJWLwl9~1p}<RR-"
    "2e9Ti4sB|^!tB4$*K!2A^<zqyMtHEL|&1~1949&FD%s1Sz<#+decr|oJcP^wg%qfZrrxE&)uNwV5v!pB50hM(;H7?R`gor79m6lE"
    "|%w6zxqv|Vdr@V6x<PLPpZ@-l8<x#ZaN!lgal=NA)Rjml4}0ZuoA-uGap#uvfmqvsX8Co=Ew5Mz+?SUP^`m+k8dd|TOsi6UE=-"
    "cWe9s9rBN}i=#gzP_|3;}9IO$;^Yuj6)k{3Kr87FbeXbOxO+e(R_Gfc?0AsC84Aon0yC+fV1PUOFA9RK>zWuV_tY59$xIeJ7m0W&"
    ")AcGel3mIo$nhr8~?dBTiI!>owKc3tn*Y5L$!J0>B!?I^y-"
    "pSKXWEw%uHuPV>mE|_M=GxXqB&`BASzvD89+)IvWD0J?Iu!LQySP2~%iOO<!7O4f*s-"
    "_n#Q2s66d_%vDsqUITh!HfJ3Oa*53T;JzAAJn+du2nrzL#Z(zwC4L}?dpl}^bP<adUwlR3t3aR@errfMkX!$%cacl0A|dEIjgI#V"
    "ep+O!oe2~e>bdjLtc^Re38p~&`^M$@nhd40*^8@lhSSz_bAW(b9&%2E|`No%FtgD9V-"
    "*lLHq2X5mG)ZIFa+AJ6FJXJ+m>)l=?$S+%!VQy2G?*Kz$kJRq>dvUqBiOkm6UWUGHSI3G;;afQHmG20^%z-"
    "fO0*cm(V0$D>xBBZ%O6-QcbvIlDiclb}HHP=+pPo}lj&S=e&p^bEjGNiBk1koEQUa2Wk>;J-"
    "W{H&A*S!rlY0c{eodr(U?j%NJ2Cv5eVe8?H&wnfC^1q1bStbFq2NOLx6OOGjVl?GsUx9lj3pAwUD+7&w<O&!~ua3!1eP<EaY~(ay"
    "btOA*G@tKNK;E$<NxpbZoPoCX4?VsHgLa3RJZmyTrB!RCpr}jT2llG3kocA_%t&Jbt7b%bjM+?sNaLwq?3f`Qlk3T<;BDu$OJB95"
    "WtKcvhkL}%rZuj{v*0L~$$#)t_mql{^TAR}C3QbV#bS{2V7aHxwG&m=O59$rzo&i6*nmOPcCqf*J-"
    "p**$s&F2JcmtS6=|%dpl9MH(##+B5CIzw_>F=PP1^XV8P7&(jQ3AL&IT6iUEMGnbTJ>lJBGdE{o9Yb?t#sKLG6m~5iXmCBO;0+Xh"
    "vzs7=V4DL2R9E>*$HqU`?LUe|8+ISte8D)w&j%EmxK22p*)4+%5Wam*DO}?zajf_1A>B@9G0~3K%&B6xyB2o&X-8FHKa51-"
    "86XN!~pwiM1l#Xc68}BU@18)^`@|f&O)%&{Cv^c)^e0(n@rnQQviDirm{gLYO3BJGHmjw;vXZqDHRDNvXS!FF&WV^;OsRqPQs1yp"
    "wxN4S3bBQu&d;49!5>Cp0%Z^Y5xrSPJ2|pQT?(x2$Qcmv>)3O|qjhU%Rj99&@4<DB@9ijAl6k(ql%sN_utXe0w74v61e=J<cl?I$"
    "XukC-zHWxdxN;c{+bT?g8Cv6m`8*J0>@b%JD3LWoz!<1VKyQJl~H7d0IvVEn?<Z2P~q_t<=qmVRohxf_K9^>!Asb*%CQvIrQuU&r"
    "1=oq*M=vqQ507OtwAa#xCfS<bp4|2}2xls|(V6KLm}IRCqM0f)R9T%%a?TlYr#5o#vWM%u#+0bN)ZOC{1Q0z1sab<krHw+~&*Vt<"
    "CDce0iR`V5>XZ(1CYj$!7Fhr?vz*Ug$MK+Sa0<fe}-"
    "dZ1Z(ewTSc<On<j6BtdWV&}MmOduf!P88`#AwCVTvWRdAnt^1+|)<|<LGL8;s>8!>~_5_3)b6kTl(AgzS_O{}^I#iekJ?gMu6L_}"
    "#c*V-IIG_NXb#-"
    "=kY{TKq3>@r9+?OHsbxx9%tK#sh(_#+l$HlcrOsMG`0N_cZdMEjK`St>^=XBdo(O(AlupXiRr98d=9tMS(lH%<h(jK$(6&C#d4*8"
    "Cakc2~C<T2CA1bl=+{Pe`mVmY~Dj?#4i8Pb&Z%tiKzx3w5avP+;W{nL0b399czyY`&a7)K#zIMJ|O+qE^Hj=rI0Q#^8x1mm5L*iX"
    "2;1~K3!eY-qjND8BeF6{dn3~du%Kz~zd_G|VGGL<?FgL6y^7Z*YoYq0BQ9w2J67Vsv%2VzLH!rK4t9jRQ_1{1A@zk(PG;o`rVIE="
    "zxkbkcAvsAFHJt@xHoVNglj!>a(QwSs$iZcVR;+-};0pCt1TZ-"
    "<jt#cy2AC+jGE*i`S!(|PX&%kug9?y@n7pyKYv{=$O)s|k{X2q)!Amdr5+K-"
    "yo*<EHqAyv9k%R;i8V#ELb;3M{cTLse6lqcB86KnkHt%|u0nJ($hG66$J+5t06XyuMJ+H*f1WC03}9yb~UUWqq3doLkH;E*-"
    "tqF9EA_dA&r`?(hvqgpoU;|lGCiNEppKpi-Vs_JD(rv8mhPhn%xTTh|9W?GkbaJJg?8(v9w;BcSC1{%{EA{xXTy$aRaU~!)z1r)j"
    "=g(g^((b~}s-<!>0W|}o^az0T{vSZ#`Kg&BW(`L598}xijFI}<F7=-R~qWP$VB3afm%-"
    "~N6M~!qPxb+wfwx`~En}PD(Nl`hK*6b2%T6)<Y;yw!wdz%B#plT$?vfc9uTSs!@vjS$i&IS;*c*QdsIoj^q(XUFVJ9;1L!BTy(Xu"
    "XZ!ic4rU>H*GNTUHo1C4s!kvSpScGLnN`iM5liXHK%^fIDZ!U(YDN$sK%k2P;_BQ5+hSd<P#s2@}Hak>A(;Ft*r&hyyVoYiyV#H+"
    "!+&2RLj6{D=F=+>D^L3k7_45$nleIy*VTF=NEIZj*~F1nv#9mb<yN_@Q8&aA<Mid^#wm-"
    "l!WVzKkGo%leUExEJ?psw9CqWIp4t=brU(Krr(3&kTq*Jd@Ec=R6d$45VVx?_X2qi?`ME?3h#)&%X&{xa9V44QE&~!-"
    "`#ebTx0_%XiQelw%D8Al5d>H4{8$>grn(UbFJ#x6(6Wsn~2y23naZ%sD}`M_So->_rM~=0xB#48Vx931}i3--b}<$dZj>5YYxESP"
    "MGM%zmD+kuucgvQg5F&-"
    "M*}>l!Re|A+BvF#TX5Y=cp?h@y^CbyQ>j21X38aZQZio|iA;^fgNJp0Rj$^#J~e+4v_1Izs5@eR}I)73AH#GBX=finxF?hM#K|y~"
    "boq5^;5kLsd^?TkhRoSy|^h0^Sk?lSS{$i}m$RmZ;R*suG_yBE_EZ{}}KCu(;#>y*p}d4Sm%=3~M!3eTgd}vtW(hXCqiSve|r-"
    "klrBT7_9sPPvCxC&0YML3-h-rYq3MAtj~PHL-"
    "s$DXmljAdYjceg*#tIa7<mDaxpkMVNkv$@#F3T^Vq$}4Z(cEm#z55v)uzoq;BHPnEn`9e?lpwYy|UN(+Gg**ygWrzuWUnVe0j{lZ"
    "?<$mU9m7SPwH&+*<QKq;pR)KlAUCBL4mZ9w7^>e;SaKL7QcYS$7WvDQ0)&L#vMH2v0ULviJ5u7hcwK6GgL+H~>OhI>mBF$y<Nhi6"
    "e2mUuY`|K>(nB{jPp_Fe)C|c)t!KKb}YQPQFRkFwb65dS|dCTZ4*NKrQ38zo5<xbB#-(-"
    "jA_B^J+}RQp<l%<2|gjvUQZ*%m~}xX?XF$`)NbY48pe=FVNcimld6zH|4JlvwbnR*wz7eN)Ek+Xg#Umt5JT`5>%Q7wgbaa%(LTIR"
    "Z#Vv-NsounP%^ipAPzl!yqX&@M<mjPzoso-"
    "e618#_^?O;Xxjo3OpskO<}(No{HpH_cs*t>CLjVFCX1wWkm9!$0vQxPKme$KI*+2xxOT0l82Nn-CC$!w3KRS7mteV9M+_LsQ}^&6"
    "lg4EPeR(L`zS_R=mv|%#uaJCHW+Z_(pzBXiVDsO5)s8Wa{gIcnZkbu3jAq(`ZT+{+wc9OM&wl@lQ6kdady$Astaro6|oD2J^v@lg"
    "S=YxIdK&y@dt1D87qoLfzu%s<-"
    "GlerQ4Q#>R~zu#$ZTKNvRS&+P#7DOdgq8!Cb+1beA{w`UbS{k5B^vnRh8CQ;qgLext^%Y}uaTYGT*CLpt}j9!RNN{52BLGG2;a&r"
    "aiPQn7@w{3{`<oRJiIrt7me8?oFL2h6X@=S=D%!&J^d9rfz>Q<Q1+6wLGryA5ElPaeg)e7Jk<-"
    "x|ZfUrd9)or6PuYY7`F^?)|lR4BA5Z^$H9=S|-cb*yl{%a`Ms=M6eS8&GH!=@g0Ny%Lb!wc(pIhnE@MsIkteBiwA-"
    "E45^`u!${6qPS|Nl^s!^<hA5N8gZBT)ty~YO{HG89lUn0<j8aNX|$mG5vD-"
    "L?8wRxmTNLL3bvyZT*n|ei!iYIfb^L6*26EPzatcCjpy1#SDq0n&o$Acjp105lE%70N!=QWr0h7+?ioohk$r~N^S&SO6r(&X*oG)"
    "@O;Egg@{YT9ws>a%<Fs>>xzmM+L_cCJd?x;pB)dWjf4}Bc7lXFIXhmcF?(^YNyBj8OtX=z<e1j|9(vof?8{DZTJba|e>$eAIU&sc"
    "*E%~&G<zPfPGP)a^=8l~PN3<uEu2#+-3Z@38kHIo)sR(MgSbZG5fN`~mLQEG`r$%I%HoQKAu0q)0{Z-"
    "y2rlS5cFN<v|z&Nv?5md`wPZp>9WFkzXFEaFx#Il)}Z$&$-"
    "!Or&;;_fE3_WhlA<408%C*9`$fu;#W4WoVS3>!TnLox#}N_1bBOMBucfXa@<m$Z;6sCi837$*9%Q*y^V7j<MG?Yqhq*+e<>g|FFq"
    "<2=2e3nz_`KY~`QeTx!gu2LG`J=rwnc;kUv3V2(;WQ8x8_`U-"
    "KI@jbF7*6W7JJ*^U;$n}y724xZOxVXI3}heKcpgsyo3T(l!+_w%%YHjY&*%0eN|n$D3HLTNfCluAm-|c}L5WrGB=KUS-"
    "%~wof;c%61I2H~h&%7H$GDA@KA@@(7IEcT@==@}?nHoKCB7_!FLbJ3=oaS4qe?Om3Y#zPxY2vAeD#MKYb&|6k#2Sz>Db2hePidp!"
    "s(aoW0PL2%8Apn|JlAMRMrm^)%WPQ;oUfK$Dt+pI;I0ks@ij8hSuEch!a75plHSd<dy)hvYDJU437WH{>oU;f6GH`qRLO|X*V0`)"
    "9f|>px3g9!wNF^xz6Ff@@f{MB)M<>^~7FnXntSg90C0|bYK=bN|WHKR+N|)hIG+q>5~f@mhYLR`DH6|1l^?}C`>@|G1nwjy^Z^Rv"
    "){`zYw*xum*8I1-6k2!2jvTJ<xPL_XJWRhjW$LhH_MVUa!lv=F`*-%vuiRfzM;9xi$pb>-"
    "m1SGCy^9+?cdQpP<{T6AwK(JSq|><O{kou@_lt)8A}{cuc=pEY9QPc4o`4sg$uE|V7<80A~e;3e#W#r7T4<0EI8R=fHn4an8SOXU"
    "$G|N8!8WzW;JnycZ>LqueUr2Ipc<(n^gug1YpZBFxTW|>0m6QYZ)w(DM9LQb{Lg%m~0TGu++vcdK7;oOWlW%Z4BM=9T?e{`!mHMJ"
    "8p?3lkGE*7<<-8^!3r}oKGg<j(VSJf4)L|jx(C-"
    "tCsc7mTU`LUZ1b6Fl`1lp+M}P{0@$XH5q88SaPafu+=?Ng^qNg?InVF^0bebDUYs)1bFhFnvEqMf){-"
    "1WrK|?M(HgPef`@o`z9KPvg`RcI#+B$UaIA6Lbu+FdpR)n8Z~0qpcV7Vy$$d3i5smJn!o9&zYkavmEOt6|CfcM%M*T#Ubu8YSfEx"
    "<ufa!;FKtfkN^UZ`d+z~498?n7C~XIKBw=Yy(H%l{w?8O^cPi&&`Gzr;A|wUBX6xgw)}|C@-"
    "4FeCxMqT=Q`!Z%adcjHLeP%D(uSk;Ci%_eY3Ul5rudR}brV0|9Ywq(ikuk#ZefcDT;;pOpj%h*?>kK71Ts-4BKdC4%5Qy^-"
    "tg=JX&h2~cxihWvUEU_)UcND|DFdKhun0(m#nj>LVu2?SpHn0?ze>C3bueloVdH_C)wDe0VpX1hmc*`xg}`v{N@zM^!pkm<<SDal"
    "a)<@fN1ngndz^gWVl05=;U0e)!(lJGTckTC4_7nS9RXg)<`3&n@QlB<J(3<L@&G*L&^k#=dH+6^C^$Er+1AK!j0LWei6+Uco2fC8"
    "GNl60O}}M+z8!MA1apsR#!j#xK%vh;rR4ry@k!|Fy6R!-?1xtpUF6wsoG_rl2qQ&&93)1z}{lZMRXB**<-"
    "sSIx(xFC>mPph5Dz_5AV4ZI;OllfObRoGb{&7EbCA?X=f{HjG=I!?P1?<o`Gl=|Gq2fooHg$p_a|UZsIHWfFo_>2<o1=?sij`x4N"
    "V=>m*H4JdAZ6{a&CaL<Do?wIzQ~m}`1Ovc{g&{%-"
    "zc{%kQP)`ARTh}i(ZXFwZ?%jFt3ZtKWWGfs*$`m^&3)6lCX2qbwYwJFL(lJ?EJw@o0827im^f-"
    "s{oi=iWKXfzoI^rIRH$dNwyFK?J&JcOn@lgwGrT}!rOi5nl(P&5p?%`-*NBn%}!ML%z-"
    "u{S7&R)z}i4X>t8S4sr&ghKYVyU#Y#i0I(i;?~co7rudcj=Y-"
    "2z0la%Yo38^U6XqCM2p~^d6_$F>v9D5Pa#YY+rN~FQW}f3t={muPDfL<kaz2u)2E|{N9q0U<7_lN2yCXUK3nGK0zmIrq<R3hC)4T"
    "Bi2FT;0lY=Y<ZXJpRWamM=G#%$NH68iF2gT*JcIVd<kJQdy{^sVvLRn;&a|JyW3B+kYyqGlTVBgx%grh6b{uFqjbdd=LR~Ar>74g"
    ")_Wj9apflNE82SBUFD&WRD&n}r<y`NR|DF&eubHjEj5rxAPP!lii*Y#SytA(qgmLs&aiuHvc?LgU(-"
    "aN6O`ces^)ngzD`_yu2wKmY3fKF#bnrDR_x(Q@-"
    "YHwTNbaB!B3b+k^h*Gs8{l0h)%?gmcH6UE!$%{9dBAFC5bVKl$<dG4!!{%8*Iija+&$4K1XsYmZ(>kdd5x_T_wbo%(o70SnYHfz`"
    "Uz>!Qfwv=h7@B3!nLAvGSgotP~8=Ys7_wdA^@xaC)<`70@KGX&AD~u3$Qyk^J=fO+&9Vdv&FD#cw<AVc;dziY)PnBA}?KSNc#*Iy"
    "a^S#xb@h3qV3FQC#A$#O=aePo@rE|j-"
    "vJSY4xxE`*&F#BN=XLANfGsT2J2gWs+BMCQn<(JrR3dJ(gSjiVD9oe`0D;gUdeY0V6oI#nT-*K`-"
    "4&u^mrfGJEU9f>Yk&FFDsqb2SA&?VcVt!9Ls|8mUm~q2gOZ?CL9p$uV#wsN_`u5(UyZ;#{YYdRBneNW>{5{rs%XUiDr)Q1~Q2;52"
    "Tom>$F{m7)>A*>DVOk1b=m2`M<@j4~_JD~(I_O@$HVxTVBfWL>%>LcS;Ht#V{@B{C$rs_to%_@fZBpCsIlQYJmD2ibSt3f@4fM9E"
    "%vx0`WkTcJPSszjQcEF8+U!?j;K81+h0wNyxBjMUHR08Ekk_VdkB4Rh~&^3)LJvK9D!<j>t!`#=XayzXZ_QzU_peJHGKpWnQ~Nq3"
    "XQcz6AOWhuZG`GllBVr)g<?~#Iw?yZ^k2nIs;^sMLoh2GsN@WtHm{g=JtjqBEj`nD)b9>p6(mh)hGR*5S#z|5gX!>6`4nP3^poku"
    "UMmyqZ@*!L*j-ej*bkR~syF%Fiq5xs|!*|Vp({7p99oa+RBo9`WVVegkMvBfa&DUvZhs2pdvP%@!ZZ&NvJCG-"
    "^e)yrsOePMGOgwpUwpR6fW4JPBAF%^Wf7m>bcn@tqUk^vb~!VGIJ+OIABhxX#X|NEc+_<#T7|M_3#@OlzLPGY)LgR2DOJleBdXkn"
    "mrvk#7_?rM@4fd|G&q(v?EvvoRRU-R2HmIqeNVw5G~906HTr4w-"
    "S{v}ZG{=I9oT@CmJ=QllLUZPRHCbZja=mmoEDkE!NT6}88x4p$!8Ne9TCxLb%ph8UYGaxKgRpojP_nil;1H<3aX8moBt9mogU9*B"
    "wCG@5xe6o}W6D!3*VFDOq%`%}U<T0?itBuF>MTUYg5KD#L&u{*mpwVQD)fng|TBIS5LbRHqGoXs!4@uAZ%%YRm|1$Mq=V~Rz73@j"
    "uu<w^APdP1nyG*+UowkbE6OeUtmyIxnDE5CGx1}I@4AXW}a|)_Hx`bf!RL2WYE%?gOyr=;A!mnyv?=)X_Lk!ac$fx?II6MAA;;Ct"
    "A3HQM!jm4eg@G>PaS-ZNuW{^56V($IYLP?cw!M(c~XOnTxY)F@kWZGA=-O)u-P$m?x2D6$S@m@UCXU~VMAMEws_~UEMxd-EEbO-H"
    "zO;VsuE*dFif3LzX04-"
    "&npoH>7N8+TOGYgn5)3J+t#816ch6J2~l3nHywN$4pBcbRWjqdx*T%%Y#68bVQ>gd#ruZ7+}9d+cY;ObI<wJsj2!Gg|p+{1D6lD!"
    "fFJ=8sHvVG-6*mDlH5!WV!6!iDZtQ~LeR0clup6ug;Ajq}=5J6J47l4q+!Yvt|#~ijcIj-"
    "m5vp2{(tLJTtgUm58OeTTWcTl8PdjveO97wk_bC~H;E6l=w{on?f-U`7?`+lco%$w`^-~acY|MMUJ-"
    "|xh%8Q2?~nHUoi;j;~_*)8~2*3x2{LB`H|+jLF1?3+6G_e%-"
    "y>ABoi8CodqOPo@@EGd%@+0)*W8}A)sfdR$Dd~FfB$16%#KU!$g>LU`y-"
    "#t>EbJ|Q&@?(=6QTHi3419F7p!?`K4&!nA2WF6U0zDJIVDm?z-"
    "5Zc>`yx>=*eg%C4NovvD#wu?EeTZ4gzikR)flAslWp8Axofqtk~3NoS^{;lrHA<3vSUQ88GDNH%VpR=%-"
    "S&2;|W`4ZQN7WEgG#SK8apY*<(XkJ@GP(4s?B!x3u{ho<Tg{xp}$TRSrGdwQ8d_e`;N~W};^NdQ-"
    "5sS<+1D4I@%lQQrS6PEcM?PiLo6#_6(zntB}5*;+U+VZFxiQjJ%9U$#E0VxT<-"
    "qL(Bp#LrZ60CziqnR2=#o>tLjP^26jOGC|Qi};aa71_`FU6x=sC`uqSbGyDF(drjr6kBA3vPizC@VJg)e4^&Sv@>k=w-"
    "QXNzeXVLv5)=2TL!qxHSvWLV0>-ajap<qH3kCT*@J?7&o7zML*+|N{87I)iXs~|Ctm%ER-yE+VNx`XEdU=qobl{gX)>-"
    "}Q#Yu+wQr{-aTp<f(h#@6Ae)9Ow!VAm>OUC`HeGw>-BX*&{?$5@j&*;Z{BQU|60&>TI-"
    "y<ss%4y~x8`ZNDDyPe$CKHfy66tV3iZ40D;1c9=<mBR%$~bGT!brI$QcM9@u|N76EBU#wscRfX-"
    "oKq^9NOB=xHXkG_#GF45h|LJBhVRmFbxpI5-QnM_KG+xA=US!D|W{j6ct-nA)20opBy*YXy+Bito9P+Vh(MbwmXh0^-"
    ";i8O7DR%xKZ?fL^5SwTI6Y?swV($f~K(8j&0(5oTl=q^X4xNu&d<BUSmxTCyjEveYoE>+_8PDD(4Y1W2lV8b;*$&abK+i+>#2x{o"
    "x&CrX6ho4R+tiX{v6YY7Y+S|*7A(dljoFLRe-6=>g>uEk}9O`E;u@7exlEkV)gBpEj=c+t+BbmERXeCwl&m6SgrV!ja{k>xn~xY-"
    "(a=nsE6J@BIYv2sgqO54XR(gB-"
    ">CWlX!XRFG(gX_cJ{d@V^kZNJ#{a?72uTCnY{I=XQrC58a=2Q34g7eu1uA?<$3^Fh@f1dylI)$4XC@*&r2ckXTWZFa8nYWB3&LrJ"
    "r{Hfi8o%K5p*|60t-*;}$J%HX;nqJ!Fd|i7{sWOc*tw-"
    "&ATKs;6(QRZ41haN^U`fw%7z9A$CHSu=$MkKFiGBW^z&8l5Rtrb8A3W<j>Ru`}Ke##Q*5Os5rNw2wGePvLosb!@7hdgt=;7cbKW_"
    "#H)#(sF!pXVOmI>&%V$8P%OosO?E1jdT-V*F)8z$+ErnS;RCK<|Y`0{`Iit6k3y>XKn*DZVJ9c+RYz-0r;szW>L=GZa33{wRlUM-"
    "Q_x3J!OUX@p%x>dKD1H6?+)l<wTnU?}w!F{TEaED)fR@p|q*Lbdj{IuG$RXT?4Z9*GZ|JGzj0@5`k5E8dZNxc(W$26j>>bBg)Yb-"
    "CN;rDNsWN4KUaipQEK(m@Nyk?a|WeGa!u#hwjai~cO?D|rKZE%c~;vMq4&=I(|kKW!yoyi{lBz5f>#>bhmhYNL|<dx!{St@8@j@I"
    "lEp~EyQJO|UCWcm9G_$=l%35RNv!y;$qm7vuvu=KsB=qP{OzYI?dU=#A|K)7R8WhR-"
    "B&A4zYUv|I0<P&qfSRD4H0Wu40zXYu0uiVw0#e;`=_asZF=4|$nN%4R|9y>H;Ii?Y@^VQ?%ge)z!(dq769QYZ4BbC<vOt6i$T&#6"
    "N86=d+4kAeAmRM3q8*U^Ex^8%Xaxr4S%3IYzct|W5u>3@t;m)La|Es-"
    "7Fu9#G=sLRT5`^+QV2rY_5xs=7X}?dcH#;cs&%@ulCOc>=gCg>ys14fK{bO?^LjZ%}Z4`#ji)FDlaP>2)K!1&jvW0kMaIQ8&82Wz"
    "m;kYGC_lk>G*?M;NZ&34B<`uP1zmND>;@o)aTZNMI14gj`Ioaa|JU0qu-"
    "+qwn_2n$7<H4s*IO?Uhtg=FOn)ZXLnN6ya*L;#w4O+D!5vO%Foe6#*4P*;eGA-"
    "!0^js3_d$eVkz{&wfW_!Og_DYQ4gRDieTC!LXWhQw94PJ}R{|yIPNac|%+iq+RY$?fnTe~i|)XQQOk0^__4!^l7d56QeOJjT|PcQ"
    "W$$9XQCcq-_p#J+bBhSN{}29M6b$kb~(Y=bZp+>Yx=oCDooQfl*A9(TeD&*0+fqriDorLPjI*FMWBsOcahg&s9Hg`lp~*5db=RVD"
    "dGL`W!gM5;ga?Uv~)j+)iW1gxBRYtr4O!?Z27bl-"
    ";J_hzl0ENqgq!PG~Ay(aFycK&KO_t74uS*!4m*wS3wcWt8U6$2UBtvWFsmKK~`cLZmyzipW5ahj8gg%Z<oFR7aafzo8%(_CFledn"
    "IpMoarwqo!ozVFGm-"
    "vgvgoT?s<w^Z7NbysFW_Yoyx=d?xenh@_~Iyp}*R9{jQ$<>OX^!Bi+MGeFzpJG5VqM0pvJo_6H65L(Cku%tz|Q&mI0-7)hwop-dC"
    "pZM}k=mjRo_H-4?$#2Errhc~3X-I1ZYVxLZhqIO&E>1ILM-"
    "{wfX7n05;JY1wDhg9u%`zhT*p4FNUyi2S1GrG3tyi+4E5MU}+Y3t^+=4nS1TTUN)S5AvO87+%LTiMxI%}3%3kMs1bOsBaGo0auEi"
    "`_jHL^R-$(z1xLJTW^+c0bZxTdAzlah%wi$xYtlP)E>{qj3X9W!<l+qB5%1m*>h^VGJp*rSveE(f*eGS<)|$E3-"
    "(5~59{Mg?y&{m%S{rRY>;Y*{w_zKr@JG<B-sn#DCI%^m>tCg#A-OLW)V0X4g88f9W*VwPse*iTE6VrT#yzwf9vBvf*aMSD_K@-"
    "v<@hEnU;O4@spstY{J7E*cw12#eBsrFk1#qV#z$j&e+h)+a6qUO9)Wnni80cA-"
    "N;ni+GrJo|2aB~~%ZxRN$HKbGYILs8oH=5jqWg2NebsXJ0QCKx_TmySUh>u}Pik5%l^C<jF9g{K8Z%Qbt{@Ec=L!L?d*evoo_YV%"
    "jrjMMvMpalB0TY4BP6CC!!|zxk!`b=*ltmifYBa!IiaeojxG|}0@-"
    "6I+SlAuA(tT2#5xF;2!p2))fxU^K($W_%cqoHgdEDrlX58mhH)-"
    "A&gl4iHE7waWcphI$v8zX`;QrDM%_@w)^9qTX_AmH=HN5I*_Z`07B2iUJay|Judb;k(HOw~pU9Q=wFdM>QBT5v^3H;Y9<DXtXcI*"
    "{fX;u<C`m0T}TS8N#J}z`Yz~9~`$8s}jGha5q;>@ta9V(`&HGBAWwBdCr7<1N#VwU>UfL`#h-"
    "|`!lG6dK@>H44;>_5AUOeGS2HREVMa1ga`<&mA4RhS`0BD$9tsv55ypn#yj660->E;E5qjE7_xSPY=TLzU517-"
    "KSKpg6K%o~*!MUzrd$x!OYkUn8Ch5`9;aR1qh_HnCR^s!O9X9V%CLfPz}Pf{a>$HGvcV8ajI#zbElry&XQ{vs9FFw7*SUbe!od$z"
    "7Hic`msQ7P)kzGZVtd*XeRpSM={lMB(eYCfAhcHduMHzu%ua9W_O3Qt00gOlKak1*NkkDyJDwEU+#jWl3M^zcb_E%T=OZZJNF`><"
    "TmQPbCJ~yJwTSMhDwFC}vthpgTFt!x<8{6Ok!haUUDH$6qh6^u6}2*PdQ&=!Htd%f!M&=ZM}y?#(z7AHwq`8k;QFdxdPFg?<uMS1"
    "-U6?UYFg7waeE4UK~~%CDWra?8i9bHBo98b?3b{2H9Ch^fxJLR_Z@CI))yZ`Ye9B+BV9nb}v|wb=%mo3?PkC-"
    "`KW@;hQ%in=F$zBu0v3TNB*q~Ls0>vrY*T3yIjjbg7R9tt>2Vz9=wHu~!wMvytOr+fPnUV|h$%!8MbH{6Mxwfe6B`R^klNI6kUkM"
    "5wBRR4)`V_m88%r&7ge!yPdvz-WR1bZyV`=s%CC(f|3tu8-(1AnrKjjF8i<C@5LfL%r->`O#%B9Chn(&kQor+*U6s-"
    "J(pUjqRkp^~If0WGj{dw3)rL*hA+FR0DyVO>U)K<=zjUpRt)iOTANWS%taWfdiw5s>GhiiLYE{<KqfqI8(JJ=4TNdpQj;QY<0la0"
    "YExau_bx5V5xoFrlQ<TUtpOZ`7O+pWJqkq`qER>o%GOc>;}r&7~T(nQ_oEacfURm;{n%3I9wiUp70dvtkDmJ95@0i|FdJ?@AnE!f"
    "xI#s#{0GW_em?&Y;(f$b6H=8}s+e>_xyT`%c11i>#TepG3u9In<#wVXg|$y{jga6#J{+;{(08|6#mAqYUG&4r>k?YGpiODt_D887"
    "9ibPJ8Zy35L=&XY{S_ZL7hH3v~GL9zOHVCAzTA!=s8%an1%pOt0)hL#g+6k><n|SJR{|ajJg(tE~bi5KdWR{$=vLNMqML@tXD4xU"
    "QvCW3WEX77s0n*<B(dOiD}3h{HiJFJ9u;oVw`%@uPb`M!Zgf3+mz{`D<&er}vC0&wZ|}u%1&vpn30er4tFP*~7oJe_)Lu7XCe%61"
    "w*>@~;**T6dzgr5g!5XNEXi0{Wc6A0F{Fql=!s%sH~rgHRKa**Qge`qc6}E$}kRD=uzr{1P2qb>YAo2bZ^JeCBvQV&WaY>Oi+ro@"
    "Ddy^=h^<M{cMUuH4CP2%t5glLVsu`n~1DmjvHFS;OH7)hIce7jQ(vf+3*sXtA^NEg{-"
    "QsC}ZwALm4TA`%;znoBJou=P<}zmG8Kel?RB{fL7%wZsz|ssVR?o5WJ7th|C{yfbrklHS=YRabtcP23Z@?M)x~PWs(g5ZkG5lZje"
    "Lfw2wv2&>#kyRx@tDH0;olkbg<?Cg{S6>KE$#RMz~i!qE|oiwk08#&(?oYt$YOgW2CVG6oL3|nG8Vte&T=szvoQ|JC32Wahc#z*q"
    "p&6UrznQfi;{bMXcr&@jMbCtL?Q5{kM+81W6X+$FQ>=?EDmSH1dV>vgjdhflxtk(X-HfnQm+1%)S*nmf_btm=L-"
    "018)akrhUO7)l>SF57MWd_G96Q&iQS?4_roa=a%)uLNq;E4;V*!bT&PREcqU0SSTD2~?Gl^nU9BM}ED*s9%pdK`QdTAliODK+eB8"
    "Y?4uUlIp6+=J*2+#yhMz5hFnQUVL#z>DfB;0s95TTBJ{*!GY|Xj-"
    "4nFmopPCHZikMj!=Ze*|SB4lI8j=_E~8S?bP4MbcOMhpfR2plV@{BoM*-"
    "JGpPv8arwvK|joN#{fna;vBXEsOQQelDp4<^><2~4Lav1NL1y%c3cA;=Z29%aWbSedq5hmr$hr)Q}SA9*Ns(KtNW!-"
    "C>ejkt~+ln>x=JmCyM%>4)w^KY>yM#c&r7Pk9ThkK7ILe3E$+paP@o}(o;wq9k!{B?ZOCbWf<yqFb3+ugmTG(FXO=~rG7t*29>Ju"
    "Y37&=61GCS*kIMFtfZEdYD3qv^d;pWmg%@_Y<e$fWQ!mKxWu2`=JY-"
    "J*GXAhYCC^NY4;*fIwpFxYZ_%FUc=)q7Yx~+IE9nz)+wJ3gohICTn9^$+yR-"
    "@5)+yqvxYA3t34<9b4xG~F&v9{C(i9fRpCMJaW2@VPZ9!hd1vpih8V@HLW#a#!!;#ur2VE;s7>>hE?>#~kOHg?<e?+_H>q@u%GNB"
    "p{NbGx#^aq5hXVQhB>u-"
    "0k#tf$t`?FXwC|?>d8T^=g)7Z4$nM@)d(rlH!gm7&{m&1&zN7(HbJR{+iZ}s8ltQx(r7UHz)XLNk9`n|bHXXU{85s%S+<sO?K9jC"
    "I6F~#BGIi?}^s44$AJU0YfciMJt#|*J(XoRiJ*@6U)hqY0z(Q`zJ_(F-cLy?oX``uK#}B9{Y-"
    "=A6!OFU8!^j!Q1{bwV1`Fq>R*F<@K`LFN`z1JVa;?d}{oTmNf3bP{zoGH}zK{R?pG&9xhA{S~B{I6H`CejSHg+3`#A={duSG-"
    "_7_>d}P*%F0XuHN;M<OW7?)OXH9lK$cm2D_Nz*BNwjQ2!%Y;i<(Lqq-I@-"
    "0k3pDIT#HysQ?N&9vD=!UO;*V%;veVUdNlbpRjG#>M*UUr(t82An#yfg1|ZFG_q8V1{&RK;njv1|}lqLS~*HvyCXN~mv8DJ7t@_M"
    "II&5eX>QnI!g(Wbr9p{?rmj*6o{k@IV*rAF9(exAF^e-2c9Zt<^Xg18N-UclLqzT-N77>R1rgA~!?-"
    "EeO(ecJ{mVYBpB}F5^pZ4MYe;7drKy+Vp&al71?eov!G8vm36M)RkM)`QwzCc056f{A{_pdpZU%s{J5&)8qF*D5TJ8itd?qv&yU3"
    "^_;CwWGkbATTwWlcJhi~y-Sk2R0S?&SpWt__d0JoO_9DdK<){hlDQ|ZSTD79-o!3<oxqU8^0+#shslki6ceP=e|5y93<-"
    "`r_ZmI&DHHf@0ityAZx9TTD?8+FBzoqEieiG*27qFAcwep%WLnxzImR}ogFY)>VI{2<9>a^E)u<D)#=2y$2I$4bw`Zm}@pt!7+tA"
    "b9m++<9W)ppehGKI#L*al(1#J<4cD-O~`Pc;Lf+hL&d}mAjdIZqXLlVw0fpdz`4xxLBW<x5e-Rx?GYEJ@Wc85JACn}^ghiiZ-oCc"
    "DV=X7+q(aydsw#bkJRr*9|Pcf1n0NO$p(uHia?n`velb^Q=9FpFF2C?>7?AM!7x*1_zd&|C$&@D-"
    "YkwdD1b4c(aUyZ>@Oce=&Wt&H#>3iB>rfW0*r)_-"
    "+d9|7|K$+%Gytuf60u;1@(z1bu4{&GJK9TX6?MMYYdOnDB>eRlMZ#0zHXEt>OP1U~y^WHs(tuNta$^OB4?c1HGg@i%dbF3M>?lqn"
    "aVp27p?ZDN4l`V<=9x?y@xxdi1bc$PgkfW^#s|6xa*bzrV?*{~h)3UkehJU-"
    ">IWr9(6Q$Xw$nQ7zLwbliD{fD=>=lmcfL7c}XR6sPnzBy)FdQH~;36EBfxs>G0%G7ZAxXM0A_=~N9*Emw6T{wez3vek{BDAb94+W"
    "t=b%{UZOtY&H)SeowoA{uKBJy2Lilhute|^<%xV=Io!x9eD^+veNoY4x^X5--"
    "+D<|fqvri~;5NNo=5QbP$hiB=FGG^JbqsD4_P+yG={go9>RG??RYb6wgCX7!c!f^d8CI6E?}0`CjAla{#6bkT-oZtgT9_^{I3D?n"
    "J@^lu&Kldq(#b>q3AQi$(&4rnI^>3Yph_xp(}lrkL)47TIeSYCm?WxuJZfZRI%6+l?*Ck4gGpp?jyzbpl8{??dnr-IN>)Ncnn}9D"
    "y62WxLZqw}=u3L-sI5mZ(`_y5JQ8=7k$2YVTXTp|XwdxMNPh-G1KXffN+wFbOaAsXuHCmsdRv5pw3oJP!|9EcNs-"
    "8ed6wzrxT4_HEK+0E4ejN)9mfGj?}JEQ@|4{Igi3!WdhkwoP0Z5ePm~SG8WVEj@_s244^7NK7xT-"
    "A=T)mY@cK`@0D2hxV(zM{m~M)Ub?a29NugaL#|tN_MN#fx4W^QF?KB^C_>+<>W0IqFMYIP3(aoubO+nPa(dh(P*4`&*mBkkLrO#="
    "E13xoRv3ndod2U|<IHlHgez1H^UR9J=BZuHWmbFj^Z?AFIM6h7p1`eC<dJV_%3>xcPdm+b*#GWUZdLP1rb?TGHk=&trDEJ{L$R$7"
    "L^VYJI`tdR1js~@fXgIn67t}xW>m!;AnKlsk>n{(E+?Q<f)&KOh1?-"
    "a8jaG5C@bTh+p+*b&sD62~OgDangk@M*GrY?DZfk``s7^O>^^s)EdQr~wYiq&W2kjNYk0}ry$JUEbY^yCBi0lG}Xp$wD8$FpeMHy"
    "k%Z@5;!q<iqx*+Gvo7})Q~o;NWYJ?ye#N^&k?$_#(YLP6q6-O_vfwXit4%dbWOLWgb9JZOHZzplNN-LE65?<V}xJzpLBSPPyI8)<"
    "G#sg?<E3yqun!4p5$di%385$+~3ZLR_x&}9&qOoAEHlUZKWENeBT7xM<N>k<BRUnIYCd=Jycq~{tCxv<MUstl~5HkBp6Xk$<zd+#"
    "I&j=DV>__I|jYno%$F;M=leb)@_l1CU#N|_iOdX^hmxRmAzTrPNF7_pv2s>GLsp>EFyz<esc$^Tk^#REPQs}K1_HBthv*F-bK*-"
    "%CT$d!c0m|=`w)ZRgrr~ut-0A<lc_p`+N(__dZ-ifd(c=f6bi(tdV-"
    "74uSS*pX3MBt8k=<KDy5u?1lm{f$;)@@6We~t@*XC1pO$I5{#b1t4)-@%rcD9PPuU-t+I?P+dnOB0?(0%QUq`QmJP-"
    "l`=vRX`+t$FB?0adWz7_%}~LlC6m!2xQK|w>V)WR$0!SEo3KUGo@Zi-IP<rd6*skua6kYc+y@?c>`Lj(-"
    "H#E#OP7pIHBt9I70c}QrOjP<cS1RjuuH0Wck_FShzy!S%;8~GZ!qT{uc*!EKMAe7SdO8*}Anco6v=T!1f>OuQIsA%8^ZQ&&ohqk`"
    "Mu3LTl~=o!@ymmmU}INjawkeD-DBb4!i2_9c#Zjac_ai3(BvAD@kVkN7E~#xK|4EyPf`Nv#))1I%#Zj~{hzdqBQCAj=F<l7Oj4N5"
    "Ol0ox0_+Po?NST02bfPApNPCHJoRqz~e7sj-"
    "(Brl$sphzkScY^cVWmq(*Pan@EIk<)L+`9*2%J+~%rH@G!|=*M$QP(*r}tqq1}V$<H@<F||&^(wk6+WDiV1bj33F7<<*&Y0mFawv"
    "ICH~$bUSMu1_%GT+WL7kMcJr;_6@)_?0GX0*>(4O-"
    "4lF+((eCzGqvz)XgEbeIPpVdc`&?2qjaXdY(I|JG<<Y<lFo2%dIQCHKBMWC!$)3#8&`dfSz@k#~I#Cu{GNV4QLIMHVxcxe3uq}f>"
    "y5}e;W%_EL>71XTe7d_ur%5Q$FBh0izl|Oi9Mgo=loB3z8F8~)Hd%*&>>uM*Ah*iitU2}&wPc}_1ut~p)dqig;m>}~^9LTT753GU"
    ";WJ0AbIl?%Kf^UbYecG=WG;j~rI{Z$rC#r>(Yi?T=6y<Et+-t`tM+Q&0LsHvQl%0Q^!#!zF^-"
    "UXF6LN5bbwR<fAGDXj%1Eej=GMIHNe$@jkW@5#o^}w`OPtb^+#v6L8#}<7tb^ISulDiVi3%_=5?E_UsjwmQJu_))CYFQ?Vpv)O0G"
    "GYBGju<{*U6#WZv&<jDx7~6)aYZJ-"
    "CGSjd?!84q;uTIeb>WKim)d)y&-+wt-%i4+lR0-Rppww<?l50n)QfFH*l<RbM6DM%R{=G_88LKqbzU>;PogY-"
    "zmtoocz1@z-<!L1NBJ--U&^XlZW{@G2I8BmzTFCPR04sV@lGXHLH4bkNfS&!8hUPu*LT~C6{D|v{6z>YX{5m?;zk-"
    "Un&%sc9)C12=&!-WK6UrAa1{VbK<RYYTp@^bsYGU%KF)B-7(o`wr95~il?q68h2oeG<OI~)tYz-Pp^bfrMAQ*ZE{O6As<JbN#tP^"
    "uPJM*r>=v+{(ecIVrgPHB_Q6e`xw{Np1D9C^W}31nAopjv9V+If<@+ukP|1#HETqpAcyN)?w2|t=}L7(o6Bn4e1@hg^|^QfgQNGq"
    "vYvKplknbG_7_#4@AUh-"
    "T557R3(k0kj^7?X^U$b?&*=MRN~FCfgb$FSeQjK7dq=bS%4=v(L8gOYNL_lG&TTI!G$Z<!T`ta6YXFkZYZ@3M$~)}lEf;A&e@Y0O"
    "nj<3B5=r1j+E7P9KB46S)IA|HGc--CnhsHfUwE|5R1DqCAjPp$p1fS<9yZ&6GQ<!@2EVrEH>#(+BFjxD@B1B<EnfXwabn9yxLDj)"
    "r`}61T~fhVuaNIy%-YgLp~^TTu~qt1y`i2(>vFlT?GX!CQq+z}npQN>vonrnBs_cLJsU^YjHzrt)N!zc`W3Lw-Cjn>JTcqepS408"
    ">Zc4y;}1+0+4URhVW9mr1=0mC$=1V&sP}fuQ_PizLEbobDw}95O}uL}YOtf7$>4ebWP!J>Tk=?DNNRo@=&rTV95?FrTG(k=_DQ8Y"
    "D_T#vR%`%&HasJ^>M;ZKJ;B$UHa4e`43Vyw0NIW{d*4J-"
    "kxevc$Y@jB)n~H8Wbm%h)RbS_6u4k`NB`Qa;5GrOuxs)sl8X=;gT>W0M9(Fq#2d(UDa^f>lrC=q>vQ7xW3s@CaJ#QXv@c>FcDMH6"
    ">uBF(F^E=c%&Zri&JVx&BerSCoyJxm?VUO}-Ij#&ygjW2wnRC0&GIvMhyMh;A9Kce4jSi;1StuV6f7++gO*~=4Q6R-"
    "33;+1)sf6q<yU}l&dM|Vjc%uZ&(jt~Q}0v^m?2ojo9_e35Qp@npt{0{9>AKHa|s`6A2nCNWSp;o^*2kcs`1;RYzSChl%<ifXYZ@-"
    "yobj4FYIyOr`hbCsV%FI_hJ}eB#3V9Ga==4BKYzT_kM^7D`~HLXFF**%yUVVgkyJ^{Z`<xUg~ZTE&8<33ZpP5*910+0kaxcfDK(Q"
    "`3?&&%-w+-Pax**-p-%u;kvLIx(^%f`U*(92{}Netm)d0{$w_axK0t<?T+r1fJ{KlK;^6gH4QP0HT^nFpC@k1S#4Xg-"
    "G?4`kM*?3DT1)@#*gyybw2kMz{QpkPc_&w7c3LDET{IdebRn17F``3>Ep?u?G8Sx*OgS}Dkm2ZBt<JI%?k8NdXKl*B8)wj=s|GW1"
    "9Um(H**fq54ARC3#pd%FC+a<9S)lnqM|Vj4*M5c4-IcEJqV@=p^b^yzIQd|%acIPUUC|YM>x+LXW1AR{)j}4*o-"
    "jI{00(F%KCDwUHb55`Ubxb>a|3^Q!R=%*omMxIR-fXtH*WVzs~yM?WkS5lqQ>j1=4COg%pGLQ@C-"
    "`q?we&s$#O;e5bkvV&;%D;IFDL979OFIGybK_BAJaIZ)|4JJxcoSi&TMd0WPO&c>1uEUG%rz8+?7aT*)wtE;-I-"
    "$JXb?cLVzgx~R2Dn>H&j_gGNXGz5Q*i~J`F!q^ko+@|ivY(}Lz{M@6uh~c>7}!qRw*fX|4n|s!R&7(J1(rEknB4z=WB0Zs$8}_B_"
    "+4;Kwc>o>eLseb29ZSyBk*7VMA?i??>_M@_nl-<v(1JV5>YHvWkyEqz1@ASfAuJdjRq6y2(+LM*08DOe1`dh-Kgm_ZzZIx#@up6y"
    "`-?DgzL90k~H}JJE=uYTDteb)9e3V!+fL^pdBPx%Gw)>7yu<J$=Vu7)dY-"
    "~rAeD+?Q_i%+5MJMd>veATw<|GBstZ&j8(Td{QL2`U6K0l6xcTI(d9N^I`t(z2x}9ZVj$Yi-zQj5;#H{AnKM|OO(YatM@pPNGo+!"
    "WC;K}D@{%lHW1=Z3)s5r5nN>n-"
    "f@QBxs)86Df{1Mx7PcB%heIB9auaJ*a!mCc=q+Mb)R}i>t}uv`T&tlo(J78>Ou?S4B+;R_O9N_WiwkBi3<++X$h#P&mnH^Zsc?y_"
    "W7FvDG>?ZXPLkl<+Pn82w*pG11-94PCAMiBTl>l~aik5%#d!Al5gbSM6pDt=d#IO<Hheb3d*#f@x`e=$$ETNXQLN%k6peFV-"
    "Z=RaddLn~?Zq?s0tpILGn8`KBFOw}<!DP#K=cYo)10J)r?D%qD0GqM7gjmDI&>Sly6o#VfcK1|o*sgyLTL$pWZZz_rLnGtT2#+>1"
    "S_q1E$*;=NmFN-e-b~*?ZrSc=A*Q3%-f&@_iK8%CxKo6L}R?O3_{3Uu@zufyGO&)7Fq^|f4!Z1PxP%bp;1sh-lmyyQBt<ofj5;_!"
    "4h8yayhD>v2;kb9LG17SCmz8b<+rTP0P<*UG1_B0Avxs?7zoIok~(_@Ef;S!I9Z0=ScrDd5o{@Jq>fx%kaf|yei|v>S@m&%=IM5("
    "auMJ-(##Ph}=-1wr)J{+G08g-"
    "cR#!yl?QmB^JN4rHDQ&6UrkrukOj}?MF^>ZRa>r?CG4vlB}QKdLXCCn)+{7^*}>Nph%tjC#Kw@=ndtPC1M?(86m*QU@eV3P$DMro"
    "d~IOUArLF(Y)rSw>0>hz?9|M6l6eJJwwEvEMRL7M+H0_iiwL`q8Y&ZULqTAv3u*T@ewdBLYv3DA%c7GJs~Nz?hCkD=4afY;~9yNP"
    "p2TEcGpNSj0#%Ae`#Kd!3wV|ijOB~v^A!fVu(*ppL#9ZcD1PArV&aQ@uHj|z5a1a@gM?kz`cyeZdMpu$<Zo7F6!jqpmr$1d}`Y(#"
    "%8b3nSwX8aRUo672DoXyE>YP6QWe?{Y@P=A3Q8U2?!iIJw_$_I(*ySgf%?&Ei_un%(?F>N%4%=T7|W*DM!OP(r#G><Kr?cy>Kk9W"
    "yqEt|FQy3#f_{oHTs+l3l{{YK}<)|<e9;)>P{jGDeb5jm!tE_97mt?b_{#{81K9a)<1Tn+9{n}7gQdtwzYdxiv`lg=FoJmy)y70i"
    "QANFaEyTaD^!UTmn!_hEwZUFBt73)%D6QYKGM|nC7~4@ADwxL@%x~6Mg;rZly?X)>x<VaCqNGY%yN9(icftCGl5>YH|#=!LY%Ctc"
    "RLceGr4RCJ7?$CAvDOy!s+O7-kM=rtYOP<p?Pu_*BH1WplOzycw&Oq#2F9%bYi=}hS=omkh^j3IEhUOi0B<Fp}%n?-"
    "(wo5TTZ3dfugl~s_-lHUA<m83v-"
    "UFk;@W3wX;=~v_%O*o7W&i>ICc3g0C~gHzN5Au;za5+}=*n=356ya_Q)vPpr6=#+0L%b2HPb<*6B`(JCVjGYhfeEFI^qH_tCgTZ("
    "|wgG^gP)bR7svq4h9V2KFKmumM1Xu;j+Y3ddY=gu6q3HeY@JGVqv#WVmp+)oL)7siWl3zbnRBw0MKF#QsZTs)*Z0bOUv0^Od>bP&"
    ";IelcN41ft52zp)=fW6z+XFLEcJX(01W3W7Ou!bz{$4BoguqM-"
    "WSEOVCHHkJ0<xp-}a$s1^2!xM1_!5RMK%WZgs`D~c6mxIRSf_0qEQb~7>0d;U^b0qm!>)0yyz!JEOQy!-%g^dMhL{yUk>|yxu-VI"
    "gp8QGr4QJQ24h)q}a?3A?sHStbFW{9M^2BSlm=*$h*Fp`KKF2fWUeOp=E+8;V<8k<+7^U>$47s*1|--"
    "m(<AB1%<O61{_kZnLn&vnoMe|`ed2rO^+GLYAZ{0t~lbeG+#^{E1+YXSNHp$?YxK>TnMz4z%jZ3TwG-HR%)3fqQRuHfAc`7Ccj6b"
    "Ny;_t9LVg}iQ=#%wy-v$qsz^Sf)*Lt<}~mA1Bw=lbNKdyLga*Vc+l`k^f+wxbt2E3nh8S)Em@`S4L-ImkaZbd&M-"
    "=<t^`+_w?)3=pCUyo%mkhEd+?L@SJZTlVHHr(?3WjuiHh?%uH5pijMK7tbM=OW7?}$KE>~f-{HTwbstLE^9*Q*f-"
    "D4leO@PvbhHb+YwuIulLvYWX^`!*r;wt)|_WBNefFvLJbQxr%l8Ky=JQurZYB0D32>?M8)lhX>3(QuJ;g9QoG5DFsQ1fW{dHw4A;"
    "|3^UvxYAA$IwX#^R}+N3#k&3dH8(ZTE$gROma3_6|3WN-"
    "^l8F3E&@Dw&`e6wdqjcfLrqKF=wd;g6PCFkd37z(em+y~@T1_AxRA9O2}<y+5n=9oAPu~@B305P9|Xht#Xz|{|;-"
    "WY$PQm`Yep6wRJ(oT)}TxMrmW*(6!J?d;s2%UZcPJF)6>`u+&?FLX=V<MJ<ldnj;<<)2>iGVl8ga>zRkGRSPp-"
    "H9GRdd8<bFxQFD4P;VOQpkZB-XKE29)ln#F6J5Glmv(BD%u?%V(Wd2-<XjFx7)fd>P#C4K+2iiJES4K-"
    "CrP0ZyJ&{3oZ;d(1@dctq!|N}oD&;77$>+~}eTQWi)Xxu{CCjtpS7f;o-"
    "w&wG@#5r(gEP&JJb$Wfb0=#2Gh1dmq8Na*O@e7*=Z@8IkAMjk4&kYnfSxXnES+>_#m%z6*2RnYhl6$aT2)``M+F{Gf!-"
    "CIn12DHfJW>K2Q$(r`YOi5MLk}-"
    "A08md7&o1;Rv&`30arBoKTA8}br=DZxmS(`&;XPFI`r6=uEGG540r(_=Y1PT0In{SDVz1@RSbCH-bNg4zzYZ`I4rblbWwupjEHn%"
    "nrqk+&agG0wI+npPOMHiY>X~Z!A!;l$FA?$XmPU_3)gQGw>N8A~Nngqfy7KaeR852q(=h8lussA#Ow{*7(=g4W?BQPdDTLPYA^h*"
    "GIR}hpQR1c7O9p_H-<fwKgBdl$z%>VX-Gh>=M^Rf@^u?<@)hv<KgX;vO~QV+gr484&Su++6Sn~C>zE=*%@L3zF?owrDRx4yXQpff"
    "Fv_~`Ph*OE%`e>^ATZqy~HD>?S=f>A)A<Zw~!s5&)OXHazBH3VJ;u~j~hJ4$1(x|1Q5Vfk6uw!TkEQS^PcB)!+@w4<c&|B|>B3OB"
    "dZoSU=j!Zm>a+>Fk212C?)r^gcTQo|@~7Kl!^jFZ|qh1$1PH?o%@WWX!o3{;Fub1H(pN+OS`U%QOR8weab*DfB;JJE+GkCa`*d$u"
    "G!51*taEY#61ZUTQht|(Cjs5nI2EGaFJPOt>1O>*m+gK#I=V9E^Od4$GNnXW4Bw4lvyVwii}$y6lR>>q2WP|`V5=i1!n`lcn34R;"
    "?sH6_CR@|;yNUJzwx1tM<UJ=SEtg9i&SukQjp=lrNU9{jJml25m-"
    ";_?QYBjFc&nC^p|OFH6Q|9nF8Q8rg2mWtZeh`%}zIp=SRRb#2b25?{#_UOHJQkUA50&=Rbv-8!Yy9_*OI-"
    "3W{UZ%a`=j{#Hbr6KTl@T@z;MzN2mb#R4eZ%4#HZ@K2kYmh=dn3{KI#jfA#E#suHKqndAxmt{mbM6k2x5}cJffW4f7F1gr#|RG!8"
    "5>QJFOnROk9W&(U#_LEjhg+hdkN71drF>NBi)38#(n;1`(E9Ys%ST8OY5uS;?0EcK(}FAlTn~$CETH^omU%CU`V5pWk4?54YeNA+"
    "vWD_ANff%*14@Gq_OZI*QjbD_K0uIG1Sb(}tb0GpU0{rT|zJ-Nu>ape*cj<Y{Y|#c{wkjY4@<?DAt^PP7=DTrd38D(|;MSdUQk>d"
    "ZSkDozq!vERLdW$MQTj}kZ42B<-"
    "ywdk3_G9D3#3Q*s74yz&#RT@@X%GeUBGs5lW&T<bL`0BadNo?n^SnBjIRcGOIBxYT@E0F=#$&GZc9fkKShIi16m!kAVuVWKl4115"
    "E8`#Y9u@N_EA$2%*D{o2F)tqWl%^XHTK*t48E*~)R$mSU3<{Sd@IjIKPLz(Eb`eUu`8P-#&?$AwK-"
    "9(o(I{r+A)P|DGnbx{!#^Bh5?%oHcGMcd9m?HF>A!dtz(p=%WwR(ml!6|!{u1W<f%`q4)bFvZ;NBhiBa87OR`D~qSMH@Msudl(!P"
    "0DZH7adhuZ_^Mbp*rca3-TbA7_EQ;gCVjcnn|Uxvpsc7O?wIs&v&E6*Qd!lX{n5*SB!9-"
    "eilo#IpS5~MP%2TYCc4cK@TOhxlxx?_N~V@ozOl<3DZfuUm`W}5KlRjIo)vC5aXpwKJ{40xMb<({aQn8?qRGpp#kT}3M7ZV!yU6z"
    "9i0mjZ6-~H^W4CU@doc%ypHW2dUwOEMOAw8Rg8(BsXoyu!tm8%B(vBv&B3d6)|rYV^vwZd9SV9qP4|F9@9@N56^mf-iDp@E$CEP`"
    "`gy1ty<b`<b~2i;eISVDOkQur*gb**ue6uvaGpm5tI80r7<1}6>p+L*5pE9Oe>Ywv3txee_%&4;Mi!`ugdEtmQ8))Buw-"
    "(wjc&hy**)ppzauatn+2gN8JzXlg=9|W1N3kd8PmPU5JY9b8{EH!ur8DJUV+TRkBF$AV(c0gvjpmCSO%^)zrt)YyEQg0fb0~dXw="
    "RxGpy)9J)|%yhQx7WogzG=8+A0U^@HulKB9!)`#oDcbRTZ8rk;|L>!<^0tL<dt2#z1fUAAOC+%pc1Kdep7E=$odCV_G#IGqS$dL|"
    "+9_6oeoMmXSk3o_D`{=)srq2pS=3xI5a82RahTvry{?oc10{TXZCP#MjvviiDfvs8>S`;>OX(>=H9GeM6k4=A8(1SiA<uw>a<o4%"
    "~oYVUMzyk1GkQ6ejIr3gN^LtDKI+9KCr9dUPy<h~g%0iAsW83}Hg1@Tg3yu#s@!ahMaj=~_UE|@Dbn7%!<a=}nl5oWc$HG=T$na@"
    "mZH_&KPpqaoEd-giL#~scywTxLqxX}WE9L}`>M^?H^c9x;4L@5bGpW<lbl)6TC=rtur%b9P;FG2cNFPlDppjd$*bun}zusr)0nY!"
    "4DBj+DeIkmfw)EaTWOr~p^sCJdJA%o7+{JO@y8Z5muKYu@63X_8l<%B<=|8z};J>DG{t*Cj|HgxJ*{tQ==P+BDTJVC{p;HmA*yQJ"
    "mNQCfAke$0V$?denUZ0amz^S!9h!}#sRezDKHsU=l)VXGVnughs~9rDTq(=p|6$3}dZE*__V7<#t4f@(YuGe62sXti|HL9fvGy$<"
    "gC$n~M&3%Y&p*8|`==U(6fZ?v)c2pD_~bZevst(koc-E-$42{-"
    "4D=W`!IL4%hm=a##LDl55{_6rJZneM0>xJxq_?mT4yz{izX$Gddukx4<206otDt9@j03TJt-"
    "YY7Bt`S}(~@=0bt4=RSE@ER{F07TUg&!=E%F`U;+Z7mKlOT?Hu|Me--9ffxsPknbW;@UakO~LVMVKZ2E2o1sfzWQODqE<^7!CJFc"
    "Bac|TGNK;Aa&xTf$=*GNt6>QnCVAzf<GOnnCO}6bL(c&p*s&cDARF=HuX)@Zq_>`7lJo&%kY_AKjkIg)lLd-"
    "0HOz;gWC}4iDG9y$kwQ6jkarmWcivz$FJPZ18~3n8Ud276M^#MBodu!yAI5b$n|D7=Ag9A&^imK%g-"
    "?{r`)>>Nl2n$6vJ}t_vRMLXOe+1|u5e<);?0}<)jWC)J=p1Nq)N^@Y+$6R>09i9CgSOL))g&UPCxUTbqNz{E7dJCgqNq?rs^AQMt"
    "4Jgg0*`d4H!>qTx|2v2pw8vM<W{DtwNLb2ynBNEZXwb=f<S4su9GDa$!SWzG8es)955wE8!adHx+*g!>=hpTA^3%okMxVbts|I3+"
    "f1Kole8`gg-X=g=<}Sfn!8kb#-<zSlCnioUL{<QBC60V&y*1Ata-Nu#}j!)!WY9eI%7`oz9od@Fc=v;fcqyR+i_$u@k9!<eMi1T;"
    "iqjnQg<#;pZ9u3)#Mwj#oyfn%l+S3J0u96M4U+*-b{PROx#L>*g#qOkOyDPX|R~!@;x-"
    "PBfy@zGijt`IGs=s>O3x>Vw$7WgDmwUV<f<(BTdbvnoZGS;b&BI$k=9KX=a64SVx%{o^}$=rPD4PBgZAD#hoc-"
    "G**jm*HaV=LT<Q<qBpn1&H?&xg-"
    "`{x5FMg?DFRshQFd8rK;!9R8I*(xiY}&Lx)TrOpJT3YoX!p7%?Z{v}cgyf0R8s`vY8N8jcwcK=u5Rf|c-RP$)UZU-"
    "c~9@d}3pZG9O8iZxoiH7@4+ctOTu>7uhX)9%iM92tOvWXpE0F!My}z97BIgG;UDmuS(B)sIxw5b<<us+VZ(TDD-zjsbA;#La0jq&"
    "2mS1zL%GQV+3TkA_UdzzW`L?Lpuk6ZH$d)4_|G_XlDeFMKn$tvcFkYvRp{B~Wp`XY`<B^`aD7&LnglV+xrl%DA<(Ge!&r!w6cOV{"
    "(Vb&cv9JEJ$UzI=xd1WHD`qe;;i}<O-"
    "I3Cxy8+kV$tXjy=1>Xc!VHjk(1DC$IUSP5h>q8@7?sp}vaMMmgMv)&zkBG}U=aFoVt!rca1e4U063^=!wk&`UA&BfHtTrT5T9`VH"
    ">ED@SX(jY@&T>UnnyTi?}49-)eg(s8i1H5i#dHcun5bPCTj`gw!ZVK|$PWak*jHIZpb7>^60M?VjFmMf9r2(kCz_)++0R9-"
    "P)RW(&dZkI=;>scO5(|C7}GV*TE5CNp9CO(gO1f1T9=TsYq(_VJQ5Vn)paP8^7kirwSW*&%2(_`aH(_Wc<)DU}$6E{o}!lk`B7Hm"
    "IDVuC&39TT7g2?tF9L0MF%lj?RkmLI0@^$z8gE*XFl*ELZ)9uZg1V?HFx?%IS&2Ey5fVGez*5d)8@vxC?0BP+x48tkAe`lh`I9a~"
    "39-j$V}489(DOF)gT+G*zK*=FqBj-"
    "&Beo;{7RZ2K5^l=dEe9EoYy(c!AD^;kB8+dOoadFln{E_R4f|B$h|>o|b)imdE42wxKs$BwktF+?lF;DdMFxV{`qk==%5u@2L=1("
    "HTN1GHuB&^Bn8J>re|9%&dSb><bBvc0xMo?F(2jAMszfK%A-#<;ngdl++%m{tC}tIcEzyes%iea2E4X#E{YR3TrkGq7W-"
    "JHoY@N2?@xN2%uQvFp`SoMc;ZUYBib*gk^D)$2(xMoavbHP>;~0J>URO%jVt>EJMLzwIPTDzH0rylM<5=`a#mEBF}A>l0t0cTeNg"
    "sWw*e{wB||s@c>6CRql*+?HwkLi%yYZFOwV*OIY5mxqSMA+72-"
    "HEJ2)8=`I&99j;ph`Nl=h}5UPeX>?45l^%9Uf!|V^!njQ)=}&mY-+=5j*S@0P_&WK=U9=t&CFu2D7yAB-"
    "&3$WKh@CIy4Z5DWLA5RXA=GT(D|Hwpp(|LF9yO*-"
    "zvBQ(Ht)UP)wdCXxy};ggY}KgXhxo?d8u=ZV3Q*>&o637syJU(=n#<q6JG(lu-Jf#LPLQTY%24qnM(S9ma}rcMh0SSyVlDbn0!3p"
    ";Dal;a&-"
    "`Yrggw`R$aJXaha*c=f*2>ho3JD#QZX93i{L9UXKQ)4r+FIYQoRn|7RIkiVCK%M4^2pkNyEoE?ZCm_dldaLBO*F$_{?t|QXq<bJt"
    "yOeqX2AHgKXq$(@f`U=7nY=q6`s5g%PQDG;nfraI4^(8rJ(l76=EeK>BnTlvQj(x~ovZUMnn`AruISC1ABsBb71+V`#2MyNc{j**"
    "q8{W0s21!}u4DSO`>Desg#0{U}i*t1LV{f{*%%SX&1U22j4WnPIMRVLEz$96OTRh!jR%`j)qf8<YTzjssL%0{}e;${}ENq@L*Mf0"
    "fI*3l{fJ?93-"
    "bWHS=8C&q`bi5gb^e&la;j}u2<<QRs76AkTxshF`;~$tLaG%<rtconxYEwyV3a7}Dlr7yqn@U4r9|=+P3>0HBikYd(#0)<_c{gny"
    "T-`i%u#b)TgeelhA_54><Sf@%36-ZHX2dKnUS%$z_LK!(MxK@5&)e;=O8Y|_U#l<vMhg<VGw@c)!GQu_8Q@-{pfNcYd~-"
    "h9s5gbtbr23nmp2J5Y^};W+p-"
    "<`O>*+kR9+GvhPau!=kto$5{{u{u;7MPi+ghWTf>S5rR;3@E`S}E^PcvWBxd|ZUdw5U5=s8`M>mtGn)OTTg;?2h|01_o+C<*XU-d"
    "aiUd>h0Bd*DI+)RgQY3!gbx04LOc&vGju)*n4IeWE?6uAm1hbBg9(`)IG7O3%&`sEnSB>VF3dW3nau!@X9ltecAH3FOSD-"
    ")A&07;pT6p)UAoaW0_<CT0pdcI6;=T9&?Q6JI#pWF4<`LYA*#<N0*@0BgN_$YDb+`sCR2<9+t?XFPDas=X&NZ8333&wXQ3&vPN9f"
    "evC@JdxhheW;V{xJ7coQ%E2xStzP5O}x#+KlonF`-"
    "&kapS!HSJCQ!AsRK!gN#5QmAcBR#HZN9&Hs2CHVb9q*<X8kF*crLCr&gOTW*)MMGpb=A4g6D;v+t&@Wx!xqE!%Ey2_{TgZ80%N;a"
    "-%_^5|?vIdQ76ougk#u*^I&8OQr((WT_|)J0z?WH2nRJ}2%cZ}EINuxkRFZU_6{G!btW(GPtRkH12m~p_AwnK$ZujIbc|dk*0c%6"
    "Euj`N4F~z{u63#n}N_1Z~#jZCogJVC8+qEC!IZX6eqt9cM-"
    "j44ZEdFRs3CosmCO_wS+`4X$`f@kwShq1r41~%iV4L1X(zY4F4S>$^+&8`{#syz!W-"
    "_Qqn}=jCm+w)%2~>GwDlR@TdQ0OvBUL$QUt4Bs?0ky{p0XHMN2Pj}rl2F9LIO#$UUdc8G<rcvW6(ylybhy5PI~$+$j@C@PRh`;lL"
    "fI<;=S${``sL(z&5BdDIaacR)%98XG(=(sA`Ixseu%(|C}qlv<)~Cy?DuVVVX;b2o$9A8Lr|O>0S|clUmKKn%5SwY{vmOg57w2>H"
    "0O(&g~w(HmXgUIeYHNJGjFzq);Xd_-c1Bx4-"
    "8JySI@Bp!e$rUIpbKJk%OOL4dn~H@d#!zB1EY>yp%C5>Lxv$ab{p67VqrqO6rubHlEzQtrSZyrSHd+JIYN7B=RRyH!r?ol#LmxZF"
    "-i%r~yZ>tu2}3$5`cT`jS?GwHme&pThUIcOd36H}#S9!oKX!9<F7@Po7=kwew;%OW&rpm#EV4RUjEFN4QvILm(HkXt*5e$}OMvo+"
    "IIha-Htls?N#&SK?we(&HL6iLjD7<0fnw!O{EkrfMh*@M@`?R~a^q|X{r#4X7&b(|O>$%0)3`AmMxh?>1aq1RLvZLtiJI9(4q!hU"
    "y)@VRp}Hq}_H<)eHx*Gby0t-OpO8id7!Uq)aBTb-Ny%;tNyI@(?9G1mRP1{aNv%fQtBzje-"
    "Rw(lhyYUBK|&Y#s%EnvKN_buw_i(FgULd!MOURnj)nf-"
    "kDdyoX(`eD>`-K|&oEFD|OYHf<Q`qP}+fVNR~OL-4wFoo~*YA0L#o_mE#uBGM&)VUVe9SG;%Ru{uh<Acb<OX-vx1Mmy7sF6Qdp<9"
    "$Ge2-"
    "HtOtvsCP)(j`*~d^bLQMsMe4aj9+oO+EeEVFA>%<51&L#tS8HG2EFn=eYYkA2Sf1QR{;QQ3(aN!Y5!ok{7k+^O|EShcZRot6?Dsn"
    "qJpnG@Vggb09|3ZP|3|O&6lTWPm-ZDby2iQyfsG*HuCL^npq`O^x^XF!}r^M`Sz5bX-"
    "^HZz=%@+Jh?v<rCnT75632K$ZKr^XYYA84ku$r7N2Ch@Lf^JRJ?J_#Bq*t&6EL@{u3j3SHGZM9BbV0$>E^$&VcP+5N6tnl)teH43"
    "a1->yrrs`c0{LB1$DouP!EGezs6PHA-rd3aI70WyE#y#|DlOZQAh;%VV9V)rBqHkMB|!Be+1QVl9S|6FOL|s=C@2E1WQpP3^+`Yh"
    "cg$r$*s6@#dGFCYh)}PQ{2sV_<Oxy7juE1%YqEYl(#JT5B(XaDsxghtqM58|t?*@eB@i0!$+QgJx-+me0Gu7nr@B{x_k1P=-"
    "GSzAo2tgDGRit#${a~3$gT9rBS_MKU|XV5x7QAC)0Si)jR;)+_UMzTY5#y>j6~R@B8}5xpZk8v%qLoH>N73vS{ZX|+!^a11Kr9xZ"
    "{gioq2{Wr7MrNShq1z#bSV=(VLPZ+@8?V0xOBLCqyM!4iZaPI+k2QeJLQ-Jk$sG`%y9_la&t|4$%2@omsq7E)Nk$89U@W&5OddB&"
    "^N0?k7>g-1VHpvQs(pqMp}XdGhhS#(6b+%5U`?o<}>U*9|={qG31uUR!E010!Q@RTGQ(dC!E%4wxb>Oa7Z<1?2d-"
    "?ihkLI&Of^tg=#}n<YCy<<_h^7@*hK?1icM_`58j5g*!;(>Y*M=I^B-"
    "NgH?Al)d+s`T!lcNS+ZKFXENa~6~6vJ<dSQ!`=gfe;9DEaFl&&AQR@W~wCgyJv9I;Y^~NY(?pM?s{7qyc`DNW_IdWj3X`IKsrbC`"
    "5*Iv}G)RSlSV}NL^(%n$&DSs&NLWkE07bENmkiV5n@g2cTPedVCov=MAYjVCksO@2vt4DR2bLq8?uwbQ?YssWlvZ6UG0oJ3C;6@b"
    "3mU1<;vhZEftkaLyZ!$G+$GtH<os<7I4Vf#C$r?Lu;ce<<M@Yzf$Sd|T1E=%M1f3>cCu|S`>C76h*R{B!&trdE1E}dwEel!v`O_d"
    "8Zt;W}o6pD98tD`k>b0_RjYZUvkca}2@oor&56yrlL8065n>taXU;Qj!-"
    "LcTDUC)xBE+gNZOOB#oiLkjDz#QvS#L2o31D865zHi%JyOXaE1i=v5?K#9{2wZEd$&e@WNSmEwG+Uz{9t}VxA32uP$$NyJuYv3&*"
    "t!g`Od$FNTO`zjzL^j<-~vs+vGEb#aaEo*ad|mu2$P5__MqRo8+0Z8$TYp-uiWO^a@yYtx{?6!w$C9Z3u5Y)4lx}2*%=<r2(-"
    "_bcu!mbibWORPVJUArtRY#0UVdTkZqXfw>eeN5EeYaeQl=$+LM@?_G@dz+4@3;;3ywP5A&QLZX%_e+WKT0CwyF@*6N5;KIt_>SB="
    "knW1_4q_*XRqO8bg8cE)(48&k4sXG|^zNU3=sC)GowqI0}G?XV(hOfr0CEhVv7@$dslIL9V#(Bek%aWwn==ELI4E1PcLeE^M9W`8"
    "52EZwauIR#d-?j>^SpAH+xOK=RzANIzcn6n|Z=eff9n0nBMNRj1_?{68wi$`&8)Xrcjqm!N^KNVUm#I>{}*Ac!eKC<4nP(YL?B-"
    "kT=%mGdqf#;U*v>h|S!yj=m8YfiKOveo08CA^udW2Zg<!&=h_C0P4EA$+v=-"
    "No^Sc30yd}&dpoF1S%1dK*NSd9aQo9VvxL4Bu}5c2XY12iT@&R7*O9)WWm_zA`@T+Qe@msf;kklPB0n)I%kw_dk1$~Mvp6VpRwnw"
    "zh*gnX-KqPq1y?-9$iS{PQ)(QXpYb)LAR09-ngJ=3o%pa4npqfN|&%|q?fv~{D<(yTwCYs+Rm_sV1yInwr*@i$b-"
    "@(@iB@9X7paxVkrR1#i)ju6mR5e3drk`ut<(Ik#?81$Up*BAs1-"
    "Kr6HCxY|{&2Vo*xNy!`hg2V)%U(oMQ2HDDQ7z42Y|eljFe8SlbR!$RGDV1<@o3jgdHK-fhTBN9mYagsd77sY%cyGRf4hy&uI{EKh"
    "T37BV&Fxx4_?uLV{S3{mX{uD;N5ozR~Ccld1kkt#{3(qn)63hIZwlHYt2^5a3F14+l(`1PB(wqy3VxQk-"
    "Do4_mV$5_&wKiN?;k}{Gb}24*F`>+1#vJ)@$~`aMjHN#~2P>U$tigt><h?H<Q>;gk=@+CgqC4l~7=KEw=!|0~;)RckV4t2lvOFK-"
    "Esx&P<JleMu;E=UvENzD(80zIqW|BC-{5G@WH5pAr-ozzI0`HlkiEz*kg*MP3FaC=lV=mZ6YK;gs!nBkKoK(Hqy-zn5Ns^<yb$*#"
    "wRrPVmD?PZuD&_M}=j01s(p4=o3iohHG~_iqJd1e}gvNwL|lF?@?2D=y=;TjIJ15_h7@ng-"
    "c~3oVJ!cu(U2O(uj77uj?l3BYR*uSQYl$dRKIybU2h;X6cAFlnS0*w7TV)e-"
    "ZWRS}%v*C5mS6=xb&_y^USxrLo%w=)Af8ZpBmNW{YG*R6X#{>TNdh&ih%%4Gf=uDEGN%h=%0>|KSto;|vAa!&A0=zm9_Qy1a9=Ag"
    "J5WwWxR&T&v&4tCY@vxhF>&sWBGLkyw%JqQ%bHO({RP3o~2$x0M3IE4i>kI;ANv?+q2RA5}0`!+1wCMt#j(cX4m-B}4-"
    "Y9J7n+7aT<u>{=-"
    "muOF))_Ev@7_&8u*OCMTwXvVWL2<$j+7_Lh?oqI7EMQc`BvF_JOV65f$I&PVyX+>j$7go7&(qI$;7J_H81>|&Fy}%f{kx3#&n*y="
    "II`3|2BV<?>0ZTk*Mzqt-C-%c{R&Yj_rx2{fuasyq@Pd3Q-{hP<FgxBfka!J`)F9wIeWAzd5=-"
    "%o#{#~z{=~$cihs<&)C9#ba&481$J!o^vYtOHHM~T_R?PHW&wH(H!M3tr*x;I!D+~dxx6;VJNZB7HN-JyGy!>`o-"
    "cKlN^2xZb_uOq+oA%xp!pdlMD>1C8!86L!%9PNW_4Gnz!5H;uXLqKiteg3Sk_EPAJ6PO`g0msx43bV$m$oGVN+^425v2hCqegkt|"
    "9!EhR&gdA=DGlqa^<Ch`toRG~;6`!l_owd7ZJEL(tBd*lTtUU8uWT>9HrcrAx8%tSOqEx`Bn#9HKtkO}A<Tq4+z+a==12oZS9#M~"
    "e;AJ*^{4S2P0UflgIV<q}<!j5$-"
    "((PV(+%%t2CHN^a(4aPgF+G!A_v_bgJE9S=L#g&e8jh}Dyj|c~o+_>|w*C0zLMXi9|F)H^UNg|3d+-"
    "uR=0YNvT^GX+AdoG9R$~EngT&kR(6DhHkH*IS0xFrs&LKN}6x7$m*g&JZBwyuLALV5!CxvbLK;x*DXh^Ks2%Wg*-"
    "fX*C7Lxn%KmvJCn3ppudwsjh~5N;Dp8)jJxq2NuTbqTb)rodK)z%zsb9BuB^L-"
    "z;5)MpKE+f1f1R+skF6`DV6UI?0FBv7DM#<4>ku#W(yFfQkKlX`5S@v<biA`sBIrsNh#<{5gO{()@x+*&%DXD{3o(7BxSFQY>@AF"
    "(gl`eRjVdXW=!a$KjSSACRP&g?nZp7}TJgw-Ll_jY?3*d<2HRAOF6K4K1|NDsf+czuFK?;{5EFMs*!-Cy5*U8M6Mi{Qkb_dM<f-"
    "RLEK{ImkrK7rhAY#|P4b2?nTvj!PY#jWW5EI!SZ)4QqWFf*lCn9i3!-107vfyZM!Ye%XBu^MX9s-"
    "R?v$i*R8I9eWW(zPhUF7Yq@oQD$#3K<7<ab}YL;UI%GbM_r*9+fCUU6gf1j$R%96MpqDZ5X@m(Fo1+(D>lgAShZwdYDV7;Q&JB;U"
    "wQ7wyPD%*VMh;g?T<U`#^+CedadKxMLQn(@t0LIn@R8yPu6Vo31J4hAhzzi(djs1}<2#Z+8~@E|DXI&0_>=rqhZ#D@yL1`oIRIqi"
    "mbE*vnfOwZ-DFSoDUWA?LL{!No_9wJi_GYX_5QbtnR_S6sqi&4X8_u+%T&T^T{5Ysx*)SlJfOI{TM@fAz0-|9oyt-"
    "E;~!ZLlDda$d*=6-dFBwK&}w>kussd-Ie!FD+(@%C+1h&zGd=U3vIj()4<GnxS1|#5kfc3P*wjKmFFab(g4QrvUWGC0rjzVcx(X@"
    "wVE%!<E~TC<Llg8^WR!>BIG)E;*|$)bmq%CR+>=-j<A#Bg{&hP5!DH^$HDC@mzaWLW?r^)EKMZK-"
    "$GpOwF5p#E^$DjdE(Mw#Ev}3u+(-"
    "ttEPLr5F&YYltE988emWe4RPIU5#}m7C~||8qOs#I`FiFjCf?~vL?Z7jU{m^j_u~St`)Ra#8=OOo{O%CWq63&guBI2TIzcErTJPp"
    "|3kww0r6L9M)@_^Q3<E>UJs-"
    "*SUjklR>vb{M}d@SwYLRRb;lydbea1mzO@<lMcU`Kk0mif<1?>ah57v4zx@08Pv5`2|MHJdZ{Pg*_Wq}Qm;0cmCwKirN}UL$|M83"
    "aRR8G2@72tn`%5W=<$`%>@QZ99&{$X3Z_aD`oa3TS%vxWwvPI_^@pisu4rpOrLpa*%Yc8;2XMRNvP&Dd!L&GrV*ViyEltHmM*4GU"
    "9X#BTnOzjSHJu7W^LACuyT;jAWRA#I7OD_pjL5ld)j#64)^FKFfgl6@U%@K8Gv=i&?=1eAwb4kOlUS5Lk?<teAT0)FzRGU%CuBD8"
    "~lZNLkV*M)Y2xM0P>6jeF!k-"
    ";=4cp{~<nS3)MKT`E>JS(^&;AOxnf<7ZpfRY%3okYMXg`R~bP&pBPZeR3xIHry|7osgJ$jX0@!HYJ^1^9d%>!y7a_2)ET?F-"
    "WTCzmmb7XJjJS-v38nzBmqa#_?7o!f-"
    "^_*bp`@%QER0>hA(0n3h3F2Zc&YOo$*dyfcQCyx@h&|`<O=~=hi7B3|6Z(6O2KW_1%pLz$=fCG`p_E}(MoU@8Ni!3&w#0kPgHJ+k"
    "omfNQ!J{85H<JaG=*St#mk9C$K?KyQ%_~VhC;i}+B;h8WZ1IZBA?S5!b!C`uVeH~_ngR@DEV3bY7xomGfvLl)dM<6qJ?f1uIY6zs"
    "R%R>mHk>ezS9wfh?Cf#Z<E`fNG$v+j{InxTKlTJgK^daSIn6@#i;iYxdStbcXm>h5T02I!O9xPIv`RWUkvrP5;lsodjJAiDcB>YJ"
    "Di!%M+UAn45q>9hR9|%Mm$y;bQO+64uRRs1gDR2Au}tmCOZ6SNn4l3aIqcbIi`s)7jE_iBh!XRBX+KJE9;r*rfiT?pXi0{s!EAza"
    "iDdP{uY~uYG#`z>#zx-L2|^w?>P+zb*>uRj>#iFJiT!4&h(5&o`olH#|6R!PSua_pvsw1V{DTkYk3Pje`oNY{z-"
    "NVgU3H0f5|V_y*ksa!(P+6MVdn|Bw%=_ypTB&U{f7P$4<w4T>M{Pom(n-Ag4}|X-"
    "~OksErBI2d2YBkvk_+}VuXcO$?JNq=xWA&a3|NN7u$o~JE65peaUHhD<4R%*_j-Xi@REPCij~D4|G4OTJg-"
    "I?<f3R&7w1E2~R6Dk(WvE6gzk5V9-"
    "w1O67xQNa4bt@)(>3UpiHa?=y>N3>$8iXEmD(Xmzep_Zl#v(AAS1&V$L~ewO*dEudqp6jMp%OZsID-"
    "f}MMbw&F((xY>fSy=DgX>hb1wXh@}>I7BVKz!@d-"
    "aLYH<Xkz(aLDopq;~}%oo}O+O~OoKO?}CXagN1Wj6u$<+Pco#ccSR(nK&|?(st4&@@-"
    "odEOR6Jz^P8Q<{eW&9BTAxj5CdjN8l&Uh85bpaUT^xS!LJCwZo6r7|XsRI);_|dCf7#<_(ycgHcfS<)W?OG1Htnq}Pj$0L-"
    "+QC4A23B(VX@$@tq+=jW^a$j#;>g2kM<dC{|-"
    "dbLaKru0PKme1#Tw<XN?V{%qfZLaO0Gtf&w;9Z@GKX{(RP%1aH5i00{FoL{?O0873?WH=RIOff28mWq#ce%Wv)h(OEw%hxZ_e<yB"
    "XmMoXDY~s1d_YO4By$u?TL&y!N7jHZwtNmIl~?%@iD~;ycyZ2kLeth<P9MI~g||LR!(8^z3ztYHjx)E{^EEdYpk15=RXA;ALC^!y"
    "EaYb0aFZ0W3B2lw#TCR8_e+~ngMfHmp(MVe*al0uB|vyfG12|F?5b8H7PF2pCZ(KXUJ>3{?s@%81HNP;Z&k5ncU6!k`ZhNrH_9D-"
    "EaFz)Yyi!`&U2+~<(lmI8%L&O+5ERe`I!Lg%U46ul-z_GzA2-U23$HKsrc}G>gafeEslI2!)`98c~`ogtgJbm{LzZ~-"
    "&eCG`Ci<~vd?;P`R^ajAAN>@^npDG1XIjsrQpA7P#ye-N;C|}p02Vtt?`P^#$eAj^yTfxU*0}tD|+|!{2$X-"
    "?>@i({`uE0pZ@Xo)w>_xzPx|;^}pV||M<htpFVu~+mG+Qe)r2SKfimv>+S8?x4cm&>aX5?_u=!G&(B8o?)knSKK%6l<MTZq-"
    "+y^~{_lrRAAVe?=U@K$;k&nYe&?t6AAb7g|NC!z_3qQ#FYiBm@>jqA@b>Y$&)+;RIe-"
    "7%*L(fyZ~poA;fKHZdH?$D=hgek&#Zs#pId*(&;EMxul@7=;oo1qd+NXI`QoqdKYsH^zs}d6Z+SkEuipLb!*}1ksTZGHWL=)u_wo"
    "5D>8s9Pz5Dj#hhO~ZxjMgo|IN2QzkTmd&rf*!>C5Yn^{s#X_M5+d{OxPc_k929?d|`(ee=uDA3uEizrT4tji3Jh?(6U0fBgLB+JA"
    "ie_RCYMe);s_+vgK`|LYGvk@vs8ee?d~^JD(@_U+$)`T4_7F30oUKL6w6m(PFx^z&c;@m#0+#FpdF|2cpAU%vCXI`4n{@%`(f{pg"
    "iEE8V+qpO^c*|4(oKeE;dY&#xEJUw;4HukU~Q^8San&*Pr?{r&Udrw_mTXx{a|{rS_!@y3381>f`w{Of=Gug^dK^zOF<(ywoyK7V"
    "+=^7Xq9zkK(npWgoQ>HWt)^?&^Qr(frHeM7x{_su__TeR=cfB*ac0`j^W00"
)
def _load_embedded():
    raw = zlib.decompress(base64.b85decode("".join(_EMBEDDED.split())))
    return json.loads(raw.decode("utf-8"))

_SPEC = _load_embedded()


def _rate(seq: Sequence[int]) -> float:
    return sum(seq) / len(seq) if seq else 0.0


def _run_length(seq: Sequence[int]) -> int:
    if not seq:
        return 0
    last = seq[-1]
    n = 0
    for x in reversed(seq):
        if x != last:
            break
        n += 1
    return n


def _alternation(seq: Sequence[int], window: int) -> float:
    if len(seq) < 2:
        return 0.0
    tail = seq[-window:]
    if len(tail) < 2:
        return 0.0
    return (sum(tail[i] != tail[i - 1] for i in range(1, len(tail)))
            / (len(tail) - 1))


def _exact_alternation(seq: Sequence[int], window: int) -> float:
    if len(seq) < window:
        return 0.0
    tail = seq[-window:]
    return float(all(tail[i] != tail[i - 1] for i in range(1, len(tail))))


def _payoff(me: int, them: int) -> int:
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


def _last_moves(seq: Sequence[int], k: int) -> list[float]:
    """Most recent first, D=+1, C=-1, missing history=0."""
    out: list[float] = []
    n = len(seq)
    for i in range(k):
        j = n - 1 - i
        out.append(0.0 if j < 0 else (1.0 if seq[j] == D else -1.0))
    return out


def extract_features(mine: Sequence[int], theirs: Sequence[int]) -> list[float]:
    """Build the 85-dimensional state vector consumed by the MLP."""
    t = len(theirs)
    if len(mine) != t:
        raise ValueError("history lengths differ")

    f: list[float] = []

    # Time / endgame.
    f += [
        t / 99.0,
        (99 - t) / 99.0,
        1.0 if t == 0 else 0.0,
        1.0 if t >= 99 else 0.0,
        1.0 if t >= 95 else 0.0,
        1.0 if t >= 90 else 0.0,
    ]

    # Recent actions, newest first.
    f += _last_moves(mine, 8)
    f += _last_moves(theirs, 8)

    # Cooperation / defection rates.
    f += [
        _rate(mine),
        _rate(theirs),
        _rate(mine[-10:]),
        _rate(theirs[-3:]),
        _rate(theirs[-5:]),
        _rate(theirs[-10:]),
        _rate(theirs[-20:]),
    ]

    # Conditional response statistics.
    n_c = n_d = d_c = d_d = 0
    for i in range(1, t):
        if mine[i - 1] == C:
            n_c += 1
            d_c += theirs[i] == D
        else:
            n_d += 1
            d_d += theirs[i] == D
    p_c_raw = d_c / n_c if n_c else 0.0
    p_d_raw = d_d / n_d if n_d else 0.0
    p_c = (d_c + 1.0) / (n_c + 2.0)
    p_d = (d_d + 1.0) / (n_d + 2.0)
    f += [
        p_c_raw,
        p_d_raw,
        p_c,
        p_d,
        p_d - p_c,
        p_d_raw - p_c_raw,
        1.0 if (n_c >= 5 and n_d >= 3) else 0.0,
        min(n_c, 20) / 20.0,
        min(n_d, 20) / 20.0,
        math.log1p(n_c) / math.log(101.0),
        math.log1p(n_d) / math.log(101.0),
    ]

    # Global evidence.
    f += [
        1.0 if D in theirs else 0.0,
        1.0 if D in mine else 0.0,
        1.0 if C in theirs else 0.0,
        sum(theirs) / 100.0,
        sum(mine) / 100.0,
        t / 99.0,
    ]

    # Runs and switching.
    f += [
        min(_run_length(theirs), 10) / 10.0,
        min(_run_length(mine), 10) / 10.0,
        _alternation(theirs, 10),
        _alternation(theirs, 20),
        _alternation(mine, 10),
        _exact_alternation(theirs, 10),
        sum(theirs[i] != theirs[i - 1]
            for i in range(1, t)) / max(1, t - 1),
        sum(mine[i] != mine[i - 1]
            for i in range(1, t)) / max(1, t - 1),
        1.0 if len(theirs) >= 5 and all(x == C for x in theirs[-5:]) else 0.0,
        1.0 if len(theirs) >= 5 and all(x == D for x in theirs[-5:]) else 0.0,
    ]

    # Last joint outcome and joint-action frequencies.
    if t:
        a, b = mine[-1], theirs[-1]
        f += [
            float(a == C and b == C),
            float(a == C and b == D),
            float(a == D and b == C),
            float(a == D and b == D),
        ]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    cc = cd = dc = dd = 0
    for a, b in zip(mine, theirs):
        if a == C:
            if b == C:
                cc += 1
            else:
                cd += 1
        elif b == C:
            dc += 1
        else:
            dd += 1
    denom = max(1, t)
    f += [cc / denom, cd / denom, dc / denom, dd / denom]

    cc10 = cd10 = dc10 = dd10 = 0
    for a, b in zip(mine[-10:], theirs[-10:]):
        if a == C:
            if b == C:
                cc10 += 1
            else:
                cd10 += 1
        elif b == C:
            dc10 += 1
        else:
            dd10 += 1
    d10 = max(1, min(10, t))
    f += [cc10 / d10, cd10 / d10, dc10 / d10, dd10 / d10]

    # Recency / first-defection timing.
    try:
        last_opp_d = t - 1 - theirs[::-1].index(D)
    except ValueError:
        last_opp_d = -1
    try:
        last_my_d = t - 1 - mine[::-1].index(D)
    except ValueError:
        last_my_d = -1
    try:
        first_opp_d = theirs.index(D)
    except ValueError:
        first_opp_d = -1
    try:
        first_my_d = mine.index(D)
    except ValueError:
        first_my_d = -1
    f += [
        (t - last_opp_d) / 100.0 if last_opp_d >= 0 else 1.0,
        (t - last_my_d) / 100.0 if last_my_d >= 0 else 1.0,
        (first_opp_d + 1) / 100.0 if first_opp_d >= 0 else 1.0,
        (first_my_d + 1) / 100.0 if first_my_d >= 0 else 1.0,
        (t - first_opp_d) / 100.0 if first_opp_d >= 0 else 0.0,
        ((first_my_d - first_opp_d) / 100.0
         if first_my_d >= 0 and first_opp_d >= 0 and first_my_d >= first_opp_d
         else 0.0),
    ]

    # Score context.
    score = sum(_payoff(a, b) for a, b in zip(mine, theirs))
    f += [
        score / 1000.0,
        (_payoff(mine[-1], theirs[-1]) / 10.0) if t else 0.0,
        (1.0 if mine[-1] == C else -1.0) if t else 0.0,
        (1.0 if theirs[-1] == D else -1.0) if t else 0.0,
        1.0 if (t >= 3 and all((mine[i] == D and theirs[i] == D)
                               for i in range(t - 3, t))) else 0.0,
    ]

    # Response in the most recent round(s).
    if t:
        f += [
            float(mine[-1] == D and theirs[-1] == C),
            float(mine[-1] == C and theirs[-1] == D),
            float(mine[-1] == D and theirs[-1] == D),
        ]
    else:
        f += [0.0, 0.0, 0.0]

    # Recent conditional response (last 10 opportunities).
    nc10 = nd10 = dc10r = dd10r = 0
    for i in range(max(1, t - 10), t):
        if mine[i - 1] == C:
            nc10 += 1
            dc10r += theirs[i] == D
        else:
            nd10 += 1
            dd10r += theirs[i] == D
    f += [
        (dc10r + 1.0) / (nc10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0)
        - (dc10r + 1.0) / (nc10 + 2.0),
    ]

    return f


class _MLP:
    """Tanh MLP with feature standardization folded into the first layer."""

    def __init__(self, spec: dict):
        mean = [float(x) for x in spec["feature_mean"]]
        std = [float(x) if abs(float(x)) > 1e-8 else 1.0
               for x in spec["feature_std"]]
        self.n_features = int(spec["n_features"])
        if len(mean) != self.n_features or len(std) != self.n_features:
            raise ValueError("normalization size mismatch")

        layers = []
        for i, layer in enumerate(spec["layers"]):
            w = [[float(x) for x in row] for row in layer["w"]]
            b = [float(x) for x in layer["b"]]
            if i == 0:
                w = [[v / std[j] for j, v in enumerate(row)] for row in w]
                b = [v - sum(wj * m for wj, m in zip(row, mean))
                     for v, row in zip(b, w)]
            layers.append((w, b))
        if layers and len(layers[0][0][0]) != self.n_features:
            raise ValueError("weight file and feature extractor disagree")
        self.layers = layers

    @staticmethod
    def _affine(x: Sequence[float], layer) -> list[float]:
        w, b = layer
        return [sum(map(mul, row, x)) + bias for row, bias in zip(w, b)]

    def logit(self, features: Sequence[float]) -> float:
        if len(features) != self.n_features:
            raise ValueError("feature count mismatch")
        x = list(features)
        for layer in self.layers[:-1]:
            x = [math.tanh(v) for v in self._affine(x, layer)]
        return self._affine(x, self.layers[-1])[0]


class _NeuralPolicy:
    """One embedded ipd-neural-policy-v1 MLP policy."""

    def __init__(self, spec: dict, seed: int):
        self.rng = random.Random(seed)
        self.safe = bool(spec.get("safe", True))
        self.temperature = float(spec.get("temperature", 1.0))
        if spec.get("activation", "tanh") != "tanh":
            raise ValueError("unsupported activation")
        self.net = _MLP(spec)

    def predict_proba(self, mine: Sequence[int],
                      theirs: Sequence[int]) -> float:
        logit = self.net.logit(extract_features(mine, theirs))
        z = max(-40.0, min(40.0, logit / max(1e-6, self.temperature)))
        return 1.0 / (1.0 + math.exp(-z))

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t >= 99:
            return D
        if self.safe and (t == 0 or D not in theirs):
            return C
        p_defect = self.predict_proba(mine, theirs)
        return D if self.rng.random() < p_defect else C

    def reset(self) -> None:
        pass

    def observe(self, mine, theirs) -> None:
        pass

class RLRepair(Player):
    """RL-Repair as a standalone Axelrod strategy.  The 85-feature MLP weights are embedded; only sample mode is supported."""

    name = "RL-Repair"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, seed: int = 20261024, mode: str = "sample",
                 name: str | None = None):
        super().__init__()
        _require_sample(mode, "RL-Repair")
        self.seed = int(seed)
        self.mode = mode
        if name is not None:
            self.name = str(name)
        self.policy = _NeuralPolicy(_SPEC, self.seed)

    def strategy(self, opponent: Player) -> Action:
        mine, theirs = _histories(self, opponent)
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = 20261024, mode: str = "sample",
          name: str | None = None, **_: object) -> RLRepair:
    return RLRepair(seed=seed, mode=mode, name=name)


__all__ = ["RLRepair", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((RLRepair(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{RLRepair.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")
