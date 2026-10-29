#!/usr/bin/env python3
"""Neural BC MLP as a standalone Axelrod strategy.  The 85-feature network and its weights are embedded; only sample mode is supported.
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
    "c-m~e?XHDOZk%^7^Ls{k`@37WA_T_3kr4rkWXQ4t!#AILisU+@AfAzYw9npay&v6dvRGC8-"
    "+%gV|N3wL_h0}0Pk;ZXfBNtL=U@KiAOGjS{q>*!^56gUpa1E<{;$9MNBh&C|McJf@vr~>KmY9?|I2^;<G=lnfBM&d`Om+Y|MGvtu"
    "l~z_ssH;=e}C7X|Mb`Y`tSerAAkM#_^W^a>%aV8`kjCN>;KXR{ps)j;eY?>|Iz>chrfJ(_w~KTJ8iT%$JfWcf7VZ1o9B#v#@zR7X"
    "Wc*RH&1Kro!#F$-`mZ<*Wa73-QG{rkEegO+Faw?{l510*Cv0vw*FcF`My5y>Sy^k?|W_NzxzAuoe@7=x`w^>ZS#9~EMHG=ZS-"
    "}&H`ZJ0n{ko)y|$-*?|izD`0}>D-d1npzIms=yMN!+RnKqgAI8<g(GSNsSD(J$_;iEw=x2HcZ9U!ETf6HHd^z74`q%p5p5yg-"
    ";<j6Fb5C7Mdv}jE_{7#nTlXD3if`zDy6^jqsq51R@-4IZtCwz7pSHfIn;m=mdvqsyw)ak5*c_w#GyD0z`F20|e51F1{O$U@we-"
    "U7yHnT8FV3UKHuc8z5>H?Lo^OA1_o2Jgx7?M_R}X9IlkR%`=vC(Xp1R&Kbqjjo<GuXFqeXF>F8s}Phky9)(p~B)4_)hC@orDw_a3"
    "bRz590kBEGF>xb>^w8do3SE7#k9bGOsSS$bOUw*1heRivw4ddX|<xb?BR?r&-"
    "ZtUYEt4}Gx~$~WiS=k<7S>+Cu8)o(PtxyLJSTA+P@dgW_r^&a2jH{R7HX?1B8weR-"
    "$Yu)5JXZCk=w>_@s>dAaVtBs{Fzs*;_Z;s7Z>ydrC#YfJ)nyy64qUkgBO5@Mp(#30uXh-"
    "N4^V{duThbqVdh~bH)7*NgdK20S?1K1EEo?pYPamw+s_)>Av~Ais+Ec6sE&NzBPp_i)ttI&0(-"
    "QJC(mii|`Oy2*MvJ%6w3WLaUFY#@O~p_3)%wS?d8B&#FBhU#@YY81s{LYp=oz;D=_Tt=^zh<72S3n5KYBk}An{eb^B(knYj5j9yb"
    "{}*pH{8j(3@<w<K?OCyw}pB`<9+f&+F8MPkqzTi_tyD9#Vu^ic?!lI+sj4k2TAddfM<i;=wrDqgn_b&(nXb9nD>6lZ~N3wR-"
    "%qwuttNqLv<BU-8(r%%)bfwy-|=&iKgA?)}DPJt+ivPmitm()EJ%=$igTuA`>aP1t-x4>UG`HWh32XcOp-M%-z7YE5rkFGC-"
    "39HG8`^&Q$9+Fe&G@A=B7_JiWbIa(xNtbc_{{pok~S~u%9UZX-"
    "~)9op!T&?8Oy#v@`ify0Xf$nJeXL_*y1T7Io1z%40V%C~d0MT{lE>(orD%RrGQc!U9vLdW~y4^l>P0vwBZ-"
    "c<A)x%=R@Xrg?Gh2$epALd}j2{7n;G$)zRU3O*+kR<5X|tV|f82BtV-"
    "o`PZcoOXt?;1*b#z^e^%ozx6>#pDM2WLwMUAT;9?p}ikf-"
    "4FfBg9${BHVFSZn&dkFc8ufVR@o1JWMV`W=oN3SunM&0*1dY<$I+#hzF6-"
    ">l<Ur@9}7%S8}Z(E2hu>F#t#S20HKHx`Csr$VNFModwta%|K3(E`#<>LT=3{4*VZU7y-@V(9s<3@}RvkJg}mbzZ#;-"
    "zJNwbA#FvTE9I`XI-"
    "Y=<<v^o#%NwbT6%hhjl~*)QNd1uPm5j`tq;uaXbSlXl{zu?bZ3Sumbe0=){y>1kH6OS>s(|@H2t+!YebQc*rGL{lR%Fn4nEeDwyi"
    "?0f`oqVYg9<ux(n96zH0c;<-"
    "pJ(s9R^F$BX2+(b7>ERcvc<itEkjXxESR6iEmE>`X(KwzX{MWB)~_p>NY3QWP3K1^yg&sFR>y{fcg<`#Xr(TFQzuI?C_P=e?#j(d"
    "E2<SBu>%Wpm3~m)luV=UWD$N9#=AyNEv#%$12K7~Z{Enu9wYobP%`ieUOn-Eu6RrdXuqz`J|;#aPgKccb~Z*30cP5)D7=MUW;d>R"
    "wWMqPk}N;v*z)|8*aF33}+AINj<c>i2YW+O|`tVUPIS$jTJQb@`1o?1#Z?zshd*&~Ex7G$<}8d(+<G6>1gtTo*bZ6oi$6==UPxH~"
    "&lrHDOziG)`cJXoU$KOg#07D_WPn#m7mccOB17&rFBSCjZev7O`9><JAgVY&nJZI5CK>T07c%ofy^Q6w>n4F3`@MIs@;_n8*Srwk"
    "X+J1d2GtrdFFSl;l7!F1F&;B2q4>Bv(=OjED_dc8V>!P=(&>UB^Pv-"
    "aPsMWnS@w^x34J{Ucq~?v3rxi8Fc^Is}zDj!YEv0Xi(mA+@Pj-XsB1shu{bK7HlSHjhYWsIrX~u}UYK_MVoX&R9Lb7s;~%u9le+U"
    "_F&MGqfo=Omz$@0BU>2Do_@thpX+Vv_t8)(+RCWQdnNOB1z<nS|mz#b>ip}?umSwTjXHkvbq*vEFnEJZ64*Gx`_Q^FDjaod7Q~3Q"
    "MQrCxt^}}u-?_({$`dTiK4EK*Yj@PAWu<?Kp{=%LW>hwkDAO%Nkr$sa<-"
    "}K;}543H+?EU<82<!d2JQqQ=}ofL{7@VA|5?bINjG)kZDvAvC4JRx;}*gPCUhkok3eML)oCRRTVB@-"
    "i_XscA?S`Jz(N$oLG8{jR36}rU#r?t{j9DZ0KzChRYXmIVg2|iuS|HO1qvl#j#B!Yd!N%m1RT*Z_6GDK83+!@hQ5rr<0AlR2iWav"
    "B=VuR{BkJc1R~hR~%A$<al*-"
    "H0l|V;%fzXhf>2$EpiqrufyL$)KpTcEv+DN^J#EYzV2Y7ji(GP?q(5873}m#+804u=YSrpZjjxp=NfB`L*sSF3vKZmw|#Zv#6Mk)"
    "HicggAx%p~4_TSk$=pwus6W!jk_KfGbSd++5R@S53B(RN`qTJG1;Lehyuzyz9HNz;WkzXb7n1^FvxIl%CwhDOB87ZKh~6_PQ|?59"
    "%=Xel@ippXQ4y<DM(3g)`^?R)(?T`QM=70)SBdB4kCiN`JbDoUJHMgC^HFkeBxJ(9$ca0dh&EK#Zj0wfeSBE7ei+0zMS<l^K$n*5"
    "oTNyrL%}tdkFHWLi_h0Ce;NBWrH3jnm4ak-"
    "8LJdg1__h?<!BLYl>f80fda@=uAlj)o>2_byXMom#_mjoT&<HlWTpomz(Z0bA;$WLDrR+odT_jd_F_bRB|PL3<N_*Dy>j&vs<4j)"
    "x#hX2YtUIm@Fma9lg+QYQx|;Ho_=BMR{hR8ei2TX6-"
    "gCTD8_V*y~v?eB@@Hh%X;oB3nX2kb_7vH{~fUcxH%p)|MFy&uc{aP!!`Y5+sEc4)%rNn2j@maNs4Y2OG^Pnx8O%hL8_F%l{M+8FH"
    "sL$>w_Rd#U2%kwiZ>SGWD)<`R37}V54xQQXbeOp$H`}2Sh+Yii&~jQRvfZ)WV=%$``0|Ku4mMv>u$EXYF>~3DHD(Om~{EFVl0?$w"
    "vaK%c>(*F%En|QCv?mqJq*;c9Fhm5AW_gcpp%Ya(wM4E$J-"
    "v6yyd0UZ=#$ZLB4uGinf0wDCrs=5PB`Csm4a`JUy2?)~uPmn@3w<j8jJq+Egpn>YfAJUNSkn(B!7WAZ80(%Nz4eqI(IU##HRfpm5"
    "pER6+p_!d>^SyFN)40edlAl7R<x6Lg)$~x_OzT3y&<c%q4=;lwLV*oA6b9FH3QRu%TlmCxwlFW^eKYc-i_pA$2ex&`B;fvCdK&j2"
    "SI7(OGB3jVY=Q^ucS5Z7r|LcdFaHY)@PhG1)aqj}GXg$rySxEIK5EF}4t1+IDqRL>e^G?Q%JJLTdcFF-+Xf2aReY;-"
    "W;4>nYllALPw274|ji?Dasp!4HdOAc~qCdKGEgt=8)3F~}W&<_Tv(@ucoufz?6`R(B;+b-"
    "7J=2*_9f*t4>{9a0tOE67^>Xxvf$qkRmrKE~HKZVLon1SX(0lkuq_tSVRwA)q)K#i1wU=9*vwV%7WT)(ERG0B`De6^S=?VMbH~68"
    "C)~^qW9&MP!Zu*?$f4WG{r#QOxmt5TCq~3`GTrFRv$NcfxnV;$a__(>j-Rp7oI(-$9bbbN*d<h5XcvT2ty($a+B5spiD2bjNJ@c-"
    "pn%MZ90s3wJDQbQtP`oQ8bhRwVK~>Etn3H}+-P8S=_9B6yP&f&L>47!pbjn)-"
    "8dD9V*WZY0s?G9cUlyd3V6xo)L##NI^dFu7+E_D^QHRX~&Zxu5VMyylIgEZ~0G!1F2&hMK!>=O>7ak-DFR*8iOns9g>fqP9S8R<p"
    "__0{^#a%ZKz!nJ*XBFi&`#aW&)+>RH2gDw!qZ8a(t3fYJ!GB~js0R<0_*nau`s@0YQ7+)(mi1Qi;rXatw+f3WQbCEeBms><czrn^"
    "duvmT1Jz~t0kokAW>Iukvy@f?99z**(SOI$r#HnJqD%~UFtG!&3Z3PPHJYiMLL1va$%UR+oD6J@3r0jf_4453*j3=-"
    "J6n8*ivmI+m6uY5h}U|EiZTjC`T$;{pLVC70{P~SXPl8rY*bSy&VgKyNS;-V97-"
    "S122#+T`639ds!mGR0E<)rDYd#p2}PAEE;fj}w(I2{Ru+?+!%@rGuwo+Fpj^rr4u#R$GY&d{PhBtV3bNfoz8+{Ouq&xx^aVIQw7@"
    "^7A*Ux?NII^g6ddCOMn-X~H|U@;D4ZiD)+4-"
    "ZHAUtdnJx$p=?Cz%o^|HgdTh$sb>Lh+bd7xIyeh7;Q$t1+CiV<6w5en8JSSuPKlU&MMVx5l3&)!uz>jLsf2z+Wx=;$P49xSfFLl?"
    "Qr}kn=>Fe0)pUcLbi+;1eb&@N>DsO0!zZf{BI#qoT>8q+2#bqL}E1-x)I>uFnD9O{_RoIRsK*!Kx0cmUL1$_nSvt5Wohb?g;-"
    "aY`CLmfH`6bGE6C83x_J>lI?pXd@QwccLBb@WdTNC4<Y)IXf!hp^98a`oPW`lL;5reLWh?I)%UbPn-cXHMmIA1IU#|3<=2wvttZc"
    "0s3r(0Q#_5og(8kw4&7;LAKGK!gXas>ilJ8R-2Jpb%v>v1@qNirvp|NDt2!L_+d`#sVY7X#}F-"
    "1D#B>>i*?+mo@`!fPB?N%1$!&0C=)t0Ve(Q6YaV0kJLxoP}kCx_UXX~?xnlZY6b-9(*Rh;R_{N;3VShc@R426O3-a*Yk`Uo{80-"
    "~x!BmNKmt7^J$4;5U@&<$cTx;;2EE^z`P89NLCa15@>L#6YmK`E2LV8huL3ntPObW@6HlXZT;PscZtep0Km`c>IuVq*Y;G=rz((9"
    "|27Sr3;zZQd3<Cv|jDS1zrIG=WohYQ}E##@KgG4tq!1`O(Z#p*{4G$%^-"
    "sh2ZuI65iUiiFG)A1fiO?9~qL?$r>&>(JsI#5@dC9baSayIBj=URbj(4bRf+Ca}BhuQ3j^F|rB?za+a5=>?L;8l^I>Zx6j3*r@^Q"
    "$!MQbsxn##Adv9J%2sMM@;xyHZfHPoCVt5>B+eh0eq?}YalMlP%=zYvMPRlJlew{5uGba%1Mh*4>B`O#XL}fUzQdrG`Wj@^XVH~="
    "1DwmV2%l*rIRKQ3F9)fgYH!|fASP0=~SAhUA$ExCxD<s@isSx=FI-y$2$cK-"
    "k%}UKuf`Mc@RWGE~8MkHz~_5ZNC+vaZsWTvS?b`5m$Ax(yXC4r*#}@2mLcv<{*3j5>`;S)S(E5=BzRjUTWjc+S4{z*_qKgA;+9F0"
    "CMvyM<T}qiQAM#kwJ5m=p}lo#vecM7OG=H_D;}+gZ}_t>XiAuECO|dZsXDCKha&IKiypnDL)0rh==AkNC#^1Z+j%*G%;ut=rL-"
    "e$JSHP!-^yn>VhJYX)BYg5>q&<c1Aav@O1n1T|NI+&-&1cBjjd(fuPnYGkgVOCu-"
    "8kaSn*gIEg+?Ix+;qL5`7oc5^W6VbDs8&E42=x=BKg^1%380Hq#S0fia(+9VT;m;eqGAE#{vwsygkmXo{>uzPZSAPN)^Szr@48zq"
    "@`)mRaiP6NNzqXDBOfpeEgnVOyeZ9<a8nO#8z!OKJ9?0a<rJFW~sqy&ao+Nu!TtEMRx8-"
    "xWmK_L`^dYv2_kVChv=o_z^isKFzG?D4W*~W&C=iQ-z#F-"
    "B2W=KHc9}p=J0E!<)SeE44(WKWxptq;5>a(B*MPgR2iZ!b1=xBc|Byy1V`RvOs3XMk{z|@c0Hi>9m1I&bBMY@-13n{s-"
    "nPaQUXreax99;(eiv$qB26cxty_8p0+d|L9Nlc^N+0PGEABI5nnsmv2==A3|$!Nc?UacMla44jhMXI|C(cpB_&lfC~SM~7(tnW_9"
    "ceU>IAlY-E8_EOnqO><)?$AM1XkK~u$5@jJFXb-"
    "O&w;aC<5RT$sT`ugu~;vI&F_256XrR4((Qw2#&1qwEwIo*mm<p&j>by`w<%z6HoyiwR^2~c6S~EfM?Ag=vWdd;?;{Q^T9Qe_ZrK|"
    "kkm&b-4xu#MLUGAqm6a(1zi8Xg-Bs54D0X#5Ck&^JKrc``22?q+vaXzh=8_6OSP}jirQ4!p(h;l8aZfZ6AS}?-"
    "(ZS+?Ns6PVHhI((X0h4yNH{<A`wl<Rs^3%?-55o{dTghm2&2DOdb(enq9iJQ?ttPqPSN8Ghr<zsG|oIaUc>#+fa)3-cVS-WD4l);"
    "ZcBh0A80DWDK^9-?NU{UkY7$>i5~~-"
    "XV4k^ARwDPolxqe>{0saF@Sm~_D#U)SUVpz*9I8SwuKWC2Rh(cgU3WZ7Zsn9a$0)*IqLK9J~C*q<!+u<Y#L7eOTq$4BUg>?96k{R"
    "zy#%w4sa)@>we|^ON@dir6n|}IkU6_-ddAvg0dnWRs(T|i=hntj>2^SXdgK^t8oVkL+ZZC3-"
    "p%ZrDck%U~xf*6po|DA(NY&iORP|b-"
    "k%6FR&L?avs^8E9VGV^!78|SYQHru*BkI$$aoxrt^&pnpR2kXQT)sz{}HS!N>B$C*SNX(uHlFNpn4u2-"
    "l<guaGjcxfzWlJ%HgPet{MkIwF4<(0z2-"
    "FEUd|0@?<~z90!DD%0IA*B+pn%V<W5(m{!dMzQEj>7zWI^qiuu(;FcPbvcok4uO<X^KZI&hF3Yh>8B`kM-"
    "$U05=|3w!|32Y`ihqlQZf89bAa_lN(do}N0bMM;s)iKj^P0n=|!EW%vw(iZVKsm6vry;NU*>??(uonI5nUwkR0k>A|aV<WLj8C(k"
    "qkVX%iX9-"
    "M^FV2S8N?Gg{F%6XF3hs>nliMyO4K*fP!o8m;3v@}5QKZNa0Pbf{0(+m939l#cf!(Je}~$58_c5nYV~1Z2SVxIT+eMVO<CA0|P?i"
    "f2D6pM(+EUNz=#qlVF9`pCqJe8M)N4KPT6g1ksZ!X@*9cqN&}>6TkOj+6X9C8YsoIrheT_yQ||-"
    "9igEmzkodZg7G8>%HfFbGmBPD-wd)Mx_KMpy~p$n9d2m2V(G{*X!(}S~*Yx-"
    "~~`gf4Mam85#?a^X;_6q<~<lhx<t;1YA;1AwUx9x}<>Up@ARwdtK!RBk%O!asm$I`_@0}J|KN?I`%}5JC#}UmfzXH0_KOIbWBE^)"
    "i*wF_nqOkN~DZX(#*WNAOn$3@&D6W*TJu&ye03}2|LjifHg`%`eyVK;DIUw%lA#9s9gx(lx(-!o!Zw6cudCtd1IuJ5S75kw9q-"
    "DvN-1Te&%L2o=9w4+T9AHdT*p(QHqjCeU6+9rfnu^oR*X1`+*~8a=H}f-3AB#jD1G4QUQ&^TOsF5xS<CDRGGrdf?1^-"
    "bgE;%)E5jS2nAs2E>ij{1`^#imYhyZ)_i;-fSo>TLzwynXv!fSTvVhA5C!JKLw+<!sVuXwrUQDQ*-"
    "$0iq&0Pr+XE?Ow+PY74Cti0{`rzw`H@FWn%r8lZ0JnI2u0vprV4%mZ9x-!oJo%k;7oQLsFBu}QV2lhm-LQz&7pIJKTkq!fJtzC_3"
    "JH~7z2oT!SdB=o5AVhg_>Q+2jn(`GS<jtJ2w!w4JM>aV{#De1Syn~J@(M0ckl=FKx>F#r7zO$CI+@aJfJOfI8M>VCzl_`+)<)(=F"
    "oubjTByc@)i~emyrq(HiZ)5d~Gz;9yi`#V`j&<`PnK>r9h9{1Q7sV2H&Oz`1PdOQytDxz8_FcawRjP$;^`is{c~TP=*E0L1=B}mn"
    ")Nc*<zx~gf;xZGJ^7yD7?-"
    "WAWxcOD>5Fs!rBkI&5z7C>WPnhc)2g$lIPIjZBvXY&7p|RYJ_$GxQ!xI)m+^qcN*p3Vq?b<M<K4V8%mqr?^ooB1L&=!<(TFec_ZZ"
    "sEo1Ejz1QfcPm+%b9Rl>V6#?*p^eMFRjw2Kr4&nbCuq$o?>{_;}k4?EcV`Uo#92J@>B!<ACj_J9L9t)k*lWY2;Ev%b2G@F<h6*6r"
    "S{bkfP%91&fIS*ikN0AFd3XYmi|J#yJ*dWy_sF@m-+-"
    "JQhzO_Ao@;F#G?cN3}ZottG{9Y$4P`h5n2Q5D#jt*T11}<f?Z$v*@CYJ%x?ObNY@mK_~s$U5JxS~ZrUAvsLC!nV5R*IIEp2Lg_GB"
    "xgz7ZYr5<aCf`*bRr$E!spr$d?(~dnWENOho2X2MFBR#Gy0V;9_6;1=%^|yiuBMa>q**2kl7;6-"
    "DT_%|_MzlYz0!DScOgZm1|g?BVkV2NTTmKH$!G&C`$IwcyWo&MEV}lJYU2^bZ0jnN4agNX%ge`&DO_0U>#I^VKNnjK$EdJA@gCn0"
    "{lB?jA~t=I-H5>equ1oOI1-"
    "qZrDa*DW^IOFRi64oZ~GQGJ5NrrV>K+fd7QHZ#nmx}frKAV!pd)7TZbB57}gQ%bK#!vt1mP~<^F&AcXq$n*w4pQ8r|YOVw{0TXEj"
    "Xr+yyW*b4cWFa5uasc%olmtaQ*qbF+(1zb67`klzV?+rpYR<XtT7h<B+}<MpP{4am0evf!&yi8V*9gVTgA7hOp>k(sH^E(KtsT8v"
    "z=K^35O^7zhu?CyJzzQFzm7U+H31&c<TvN(bKwv{ol+Xm%CyGlfdLj!n^Jm??EL_JpJB41f>9w2%JH;O_55g<`Bn@?k~nR8MU^4a"
    "vvhLt(HM$0*asf25OL`iX}5;lW&jZq?JiWm6AeW;c-"
    "zsALLqQn#8V&z)EaSvS_@hLda96GDrJQre@P*bMC}3)7u_y)jjE1>NE#Xx-krf_`jf#%Cze=bKpB_03fo6P)}Obtb;LmicbzuZ+P"
    "Rb$r#sD=VZvkyf?ng{P*oQ#+aiE_&D6Y!QeV>XNefjmJ|WTuu?vCGh+q?+NZzFt!2w2klDto9@n?cWFQl1u&+(wxug%>Ol<uZ|lY"
    "^p|**m%YASKV7YtoQejCL|f`t#)f*;Y0&9Oag~dn-"
    "9woD|nDC42LJ*A0m4a$eLMBJ+diNxgD{OQ;eEa6$oBq2NYeSR0<S7Uj?#iBwufqlXJNtmS>GakE(0DyKdYz)B31_Rz+gupuLwqVZ"
    "Z_BZ^%e`B31}8@L5}BPCx{3Jq+Roq-bJm7+#3O(_lh0@NVq_lPOgP)c#g_iQeF6#RO~=ulE+2ZNV9%{<|gP$65}OPb?8X=i51kWX"
    "=XwbTJn_^LJsw~R)qpT~yI#UXSdu1AriGLgV*24~e7sK5}6@DVhQ6Z)g*+ORNRl5n9)PS?MB^|vRf^N`(k7h(oAQrbn!7=pJ#Pl}"
    "1?ijbE!vWa9!TWxgcv3Fiumi@XSA}j5|K76o&esgRPJt>=V33n5zQRF;$XEP5rKzbctkU@+FM6K&)$)I=j0wIVl{sF5;Nf>D`crS"
    "$?Z-ZV;92@gNxb6=1vH2$LsRzMkxY-%m&ZH05DbwN1rs9BD3p-YITb70#Y}4i01-"
    "?r**J@6EZ(u9M9aX+YJ=MU5GgLQFN)n^V**$)k1C1c96{z#k@zO$q<Ofn`)=0D+e3}Mx_)Ex;_X(&tX_ovHL1gcjKIegwdrNV6ED"
    "N$c=Eq0%<>>eOtBNT|*_*IRv+Wc$NuLVl0f-"
    "5_;e3~<{IM>OVP>CS>6k%aG=pDooK=s~>L&K?l&5l@3^?@<g4W4n3!M+u0X+0kq6$?u3jxJpr=&oYE!ud(26oU#MzY-"
    "9!rm|<L*>&sN$46viz~GrJ!=jtTAEp3&|Qb4)2*lLKNRx1e$=7x@|#y2)X;ZGnu;NfrUC8<_++FkAM2Z%45)KuRR;-"
    "4Im_U^sUpo)VTv!E2}Cg2&EJ;(=|g!W83sN#Zl`$<6rFjz8Opnc_j_bulnp)yIK3=m%>@muO+gKivB!a_bDyS_Weht@;e}`j4VBV"
    "@-RoP3mDimj8Z(h#nlCswNmP0ElCdfoQK|)6YNk(Q#}fph69{0L#i?W3)G<;7Ev+;jrlf++(dhb-Tn)idQxo8lJ<{Mlo#t?>3lCU"
    "CTMMG8O3FoW%j14HMw=!cs5f9HykJ`201_p}qg!#wrx91gWfDg&X)a{8N-"
    "I#l4cftz^lqg1qglQTul8|!k=||Ck~C20Ta2ud+$ZJm<A*n{^8<xzbTKcT$>=gc(FIL}KQUJ|_=pd*C+I<v4rHZ^!e)TvlUzk6Kd"
    "CHQ&~1yLrNq#L=}1mWuzO7Hvoa^UgUuutYP1D&{r8GUvFRLbBZTqTr^P;O)-"
    "MltT55U`$kYIPiooIYUT2B|T=nLjxe|y5Y3QaJp}CKy6U$8IkyYrVs@$-"
    "#2FjaA$+wy9<|pX1h=UYiEmbF^U;x4h^)M@SMbp%Q1Tl=R>vW{3&*w|APyZI|zz;alEXV+0KvaY&l2HuN`iD%f!oB!+|3BSM-"
    "Q4H;oIu<ILZp?rEcKt%+mR(Lwl2)rNpw#o4SOHp@IbfBMkRDl3r8;+!%W7$=;eLXrk$=~HPM$T{Q93$xn#GD8VV#Dz8jo8a>9j<g"
    "L5vzrV;B&HuwNmP(<hBP_hReb_{gRM0ymlCPKRhl{+EX%?me`^3-$hh(U=(u(G}vZ|!4`-"
    "p`+>EwB>YYi9Lydhei8rS;=AaKtr&YM$^HrXPtX)Z-VhD9}iwp*k(^t7dp-"
    "QS(rH75fLJgX(oJ#H4!bW(Yb6xo->PLXnsNenFtd%~tF{zXm+MkOa)=%T4nb-tZ=Bqi8!c_CEygmnt!O2aCYy!c_V2s*j|Ham})<"
    "J&x1F4LgHiV*DimcVJgo?qB^m`gaPI4++<cn(e^ppm_zQ{}y!&O5Wl@Lcz#7h0X`;41kZB;2aHjg-fJXJh@y4ADN4G`U(2-"
    "NJiJBrD%oQ!v`RxG7WZ!1urc+TI5vR49L|^^}W&_&JbwRfD+$H60lYrYQ6pIRGAKpVh+({%gV)MsSXE(6g<n)1?v&+RVRD=ijPAM"
    "Nv>|J{f?A>As6Z3GeA5+wxUFU<)jrnC>>X#x+?Ot^zB0dT5ytfgz*BR)oYG<9u630T(GDGNFJ;Rm=?5DkOpM|q9@k@wt%yta1~b)"
    "@(6X84un_eFNw(^E7eO<tSK`GxFiq!)$unpy=drY|FD#9T7pnR6S`Qlco6)#7^dt^J(HP=N;(h@(t{p;*&%RD3}dpcxPlnph^VHY"
    "a#@KnKVmkI4Z(LZun!26)7qVcv<dL*KzJ!#Pbu7y`T$fnS|F8RTcg=Xm5Tz&BS1{)5-&0D55yNffU0|-"
    ";gvcksM4zc<8aDmfgvwS3Kl}LR-RE=(sOFp$OpRBYsXsudjtEW2WA=|Dv)e35;UWQ0dpo^0|cB2Y7M`>PC84>VTf#e1uEM+e7trR"
    ")Reev5VOxsn`c31QoS1uLt3WVytxEG8*E2HZ|o-"
    "17mN|18hj?s+F(U&D}Eqr&uky)Xua?b&6t1drtTcE@G;>92~^;}0C>$V0k>sUVe$+oi?!S{m3Z6LSjgCYXm?B)j*h*%*$ioHLel}"
    "4lWn0$0Mya_p#;Rmq9ysFv2+L<W9%Z<I#{viU?ASY|BqW16ag0oPMn>%xxmrc?(k&58(@lbnxGEliJ-rwQO{AcgT4YiM|%Yg8NhO"
    "IC_vz0Lcme`s)G+m40fWNvwcCK0ER#o4|Yr6HPCUWrlXsL#pfpS+^oKjK3mUXb=i%DPK^F3`_Ayj9{Oj+)Vop&rR@)+3|ug~=-"
    "3qjL>+8eT2^+43f?!eQ3F>@qp*a2h6W=ONi0&pTd@^LP?3cRPlRS;hv*)!3n37u0|ffGLR|^?3gl{UhWy!vsRc7wX>^oL{NkZb)s"
    "e#a=qE<oX`4pWbK&V!16`V?VAcU_0*uWTAj1~IIX*Lgx(U?1$O8(M!$uPg;5ul0-g2a&)Y7G7Af(gHINQf-%2yTqCsad)*pq-QB?"
    "_c-"
    "2pl9)`90168g3yAHeS*&;6rwbh2+OXj)Rp4Sw9$gP(y*bxT_5wMI$vO30lV+tR5y;W_X)$L2nW?xxGs1r&$Vsn5&h50J<9}w>1z6"
    "AR^q|VDBGpI{`O??`p5pk3ESQ0P*mKwf%@fY3^Z#It1m5bUQR8NDWP2Q9#tcyh&WA0fmng6011Z_&vhop((lZYruQmJcVsVKALYB"
    "40j*Y_O#u0a<Y+!;(icR{3HfK$VTq8ZZ+(o?eb>}Qe5+<Am4(uW1FAq?vcGyAFviq=}QL>oD@XwC$d28TDBe3Lvr!VUpe4b$ZtXU"
    ")9H#%KPE~PHzBiR)$<j$ze++aRU-c<C66L%K5^ILID0_+8X45+{drA+yCrj+BB7Z}O5{G}r~!3VU%(?pWt=1uT!85Pe-"
    "=|0;~fhg7={NImr4UElL48q)`*s&eNq;j^3&r%BBy9pQ-"
    "ICX*@7(xdC`E-kUkY;nONfhjqjC`uK`aDjSM~;f$`W+01(+)*Ahhqz2>+|r7@Gt17pXR-J-hOd>SBxXQdiQe+IVAsvXg-"
    "K>O}re7Li8l2{7lL{}fLmM<E}dlQSI@&tE3-X=2*;pm4ENmM|=J?dX}2_EhT=y#|A%T7iwkkUd|uR!-"
    "BPFT2ZNrkO~oJU>vb>NU9_KO!}<doe*S1`IqpdFX9ZZ#TE>3)R^1E=prk&vjQoev<~5DWk4k2EWP^+TxdH&bnj4l|07jYsaTQ`y9"
    "E9b`~Qcc{`4Edh+%N-Ps)q%B<gn=W)CPvJZ<e;LF+nXV95AfHiM?fJ(kPX2~11Mlg}D|(bC52vbIqA)Nn-Qr}eDln964xt&(69Q~"
    "<KSA+;uxPKqpN&s6(=7C7J?p4%xuj6qV#ayK@Q$PzyaeJ~0j0p$SC5`{|5xT1fD(|SyTy+!8=@DI({WPeE>Uv37Q9K5QC|v>QAl*"
    "+KgKXsv0ZmWP3FKuAr?1k=7p=pN`90*S(eG4a3{{s-"
    "h|ly3_C>0gdxIbMKyjPaWyDef#^WBd(wz?`ft}9yzws&6kP&dCooe;PNO{mwuGet4dB%(4sJ?h@-4(Vp-"
    "1w;hw6@&TGA#veCY*jM;%i9A|OhVd|l<8b->{v<fH-KT~L3a!EN;ASlpc~_6LUqD28Hxbd_n&;~)W?fn`}B6&;4^1Mu#24#Po(Mk"
    "z%Nsdz#v9d3Nb6HDJQioWQb@o`pna+Mab1Zh3)LF|eb%t?Kek^qu6O23x~5v`aFn%d1O2}FId<I(p5vZuoKSbT#03J<y$5XD=Z<T"
    "VfhZ^3PNsL&PyEiD^9@&=H8sE^r(20F#8<8ku6FZ&x5TCNCj_Pb?yM}u|1zrN<lr=25}9}>pUV&%@`OKe$ie9%PEd<pa&oQm=xsq"
    "80Q5t?=A#hF6X1%S@6kcYs6V~u1SI^qdi7+&l|>xGf!_yMAZ>UFkLy5H)TFU}jRVOZ)}caj@|ZsAfA=ks;~3&{WzzJxD6Ojn!^;;"
    "fk*dgyF)J9%|Uv*1vf&KZfVC%qaPB8_bk1&$NR=zs<_$r*uTQdgmaZ_`942(QdUG2YUngMILUh^MFrYbd)w`fQ9N0b2sv##ZJQT("
    "1YX+-=InXsNQ%6M@+&_s#qmGn5YJ^H9U~FvII`2`;REMd;!|LG-t!&9a?iY~JY{7@`Yuuu>$E$-"
    "&Ph%PdG@fELoz<oyQF3D+D%2nyHU>E8*Za+<kdW2Yb{K17`u-5D*9q5w*W(j{!2N|9!|A-"
    "+;n;g|%@%uJRaw@#^7j)ttkOW3yPGO7v-vuW%*=tY6VhwF*qV@YqS=wE|$8hqKB@A`V+Zt7kgeT0A{kvyn`>?_-"
    "3oOr@$r)bN=CCNNODFo&JhRe}s+V3DY{4Bi3vUR6Uj<w~=VSq~}ph>6va42c_pQz8U3DoKUMcg2NxmENgmW5{br9mE9tACr`W~Sg"
    "ba(!TD_)5QBO8*aZ3cXZ0#K3!C9eItVegO9c*JxyQcW0H;;CVvWfpZYfO0e4?RCO2?uiPS*Jx;qhg1EUxZHavOA$Wpg9&e~oC_<E"
    "kV!Dg!4w3oT8;5c?;V$~yv@h`Tt6<pC<xrb-"
    "o9(8x5?wOjUII`Yksnoe!@A+KS3;U>U_=Qc8zDg0q3RAV{RI~v=^DUQ7*>Rh_R3fXe=-"
    "D%+xnV3vuy!zu*>RL@T9V4!Q*q;s5EKZYU$Z~)W%ouR0Ee}AEpdqpX&FxV8RFo%fOz3BKyqgV!vbhLiq1)pR)m*r3#tO1!s``#!Q"
    "U~8y?u#o10y%#&F$2PXO(D+hysCx9*Lu*KXRzEb~X;5~#_X8_Ks#PotvZ3`JU%CUh51U2|4(04Z)9dNWv^@DwaRil95#XK=Y6AZ@"
    "B!u;H=+1BPu`N^TMX8u3m}(kOU42$Fq*Zl}{PYCYltbh@Hyenjz1wa9N~N_h0(@N*;~#{3}<VDJmXJ`gIi&j`ichR;ZQ%BxfXRQu"
    "V*_-IAijTYGZoFpa&#K+scK&Qa^istQl0Thz-v-"
    "<YahI{t<>j_msb+(*{Y7?X_nn^f2q5K6>e!mugLhqV56TM($kS>EX@#tqINY_MZuxjw)?DKUei0x;fdJ1GC3U<3e|BhO10P^bwAu"
    "tJE=oAlHnqj~-9Y`)v0*&&8<}dh^)OjG{FxQe51nyw&p27ee|45`rX#ufdZ>+K-"
    "KIT=>72k#a<C8^8Z}Fe9P+LA%j`jV!rT4`qY>nV7XN$zjLADM_6ixJ?20xTtba7X*Bw>{gG(VsS_xy-Ywkbp;ILh{Eng{?Biu62d"
    "PF#Md<{$-"
    "}R=%XBl|MXSVCK{su&WKuVHiWFxiT_m$VR3MB@c0e8ZRup@KJtQcTIqn=Td3^NC(?y7&fN(!V}556acBuO42XG9%d8fK_TnArb@y"
    "G!x#a#$z<Tbv?0Oay`;R38cy*7W+X(?WM5$G!}&p%AQU8KWt#?pn`TcS4_A|Ci!u69`NXL17g9?*?8K|47N?Ou0c7*$pt~Tc{~-"
    "Sr)?-"
    "_T7c$IDB?01SPE&^=C@%3E)J3q^g(bpQUIfZN4j3C`lTrExH0^kF*uJgsxrXBl!5^pCYV=qv!Qo83Qzp~yp%y_e?Bv0%fYJ_YTDK"
    "XOzo1&0el#*kWT8Dnpk}zn<FGZgk=#FwA+ca`09%lec3C-"
    "n(bQ}PeFEb**;wxyQ;F}Sfdt;hV64Xm&`IB!Ht!+!A2gsPeE}n*@Ore6d%kj9MbEN>@0S^-"
    "xy8J<1z|V6E?UqFH{2x~J(UYNs2Y@QV7=;@;0Fs6en5UI`@>0#gM({6TfyZqg?eNgtkN4j;ap;h&BYfU0!BHXST$pT6?UDk9Bns3"
    "6At5Lwaw4B8B?*ZXzspN8!EsufH53x60ntK1sl4Ao(5Ug>%(osA0&JU$cpF|0}lm#4nl0_gtQLhnm0}V5ud{g3woq~%%uy7?up|("
    "k{H8mmxP+v|AmhZiq{21?FMRGUS}K7!0_p{l3#sHz|nRa$w)%JZS5e%L-an;D`ExfpuV$dwh$^!$geo-Mzg0GT!vi_0Q4nj292)U"
    "{tb!d{z)*Qh&NJ8$S8S)(#a;=K@>?NEVyhmW`J92lj6)m#zmI9XV?;wC)pge?4Dx%;3);~OEuF;W<NCX4~FSM%3OuUVd?^8Ns_{q"
    "%sBu9QW|p>lPtr}u=1PJ?40<aWz33Tz~g^}egFz;;QRd=8VHz7L}XtwiXu9Y@GI@(4NR9n9Ex|IC!h`8m*_J*cKoxwfAIrbvu*(~"
    "1s@b@ifK=8LjW*<V&~(yJS-W-pDOD`Xn>%t$Tlmu9@m?1+aWxe{IEz++Am0oJ#(RE3LhoWL|?p;Z;RS@b8^tZ$Y5_zDBQJ;oQ+a#"
    "xak3W#1&V5Mzsn17;H$t`u=VpH6o(`B-"
    "s5Q{}Q~7g%>4)DX6$7?+6QdTHSFL>@PL%VC`<8Tr;Z!?6l<r?RvAney%~Z!f|T!w1k^}E#xxO12(D;3Od~F97{`S72Su7>A98k71"
    "wj)-++SBYa$x`2H2NL-"
    "Q<x~{G$a!fWfj!^S$Ss9FJYV$Sm7*fy7Ij3x4faVgYAg&d{0Rk|r8^NZ$d3yOXruZEA}~dB3DmLL>qT#wmOM4Z(z@Vr{psTJquuI"
    "(}#n152z+^OHEZP}A-"
    "rrEX%>CAUPma!TTct^!yZfU+<Sk`$ssH|Ujv4s7m%_nl=Gq5p1R2L_l4nWxM=;q5)z12m?>4<D8)XfCG%kbYGQWTI&7(8!i}0>S_"
    "eP|LR0)il6xgz^hNm{;6r3@z=@*-X2RScOFFZ~E*(d5g*f=Z>CBiB|Q6h(HqeeWj;!3y_HidNk=`1Y*7HHW-HQoVNf-g9K-"
    "#fij)dFXlKPWXb5W?F0k4y4!HXxGq+DpPPSrK`-6~1Vl{A-"
    "7oH0J|KBpCRxzmh)fUHEtp<(4O+<>X;e^^!<<+k@*Fla4q<vStR{RdtawMvcWI&rVbadGtbl>ZO8Pn;58Ob554kn7m=Btz{fU{!k"
    "V;`aP?LNPE*rCKXb~Fmc^kj?I2YImkfbTW6Bc^Y_6S7^4Hf@=?zcL&RwIWt_i_1;aZsXmgo}q(5rkl+|09vi=6cISqoj$VBQREv7"
    "j4>v^HTBfUfssQv7M}kSu6tEIYw#?qyouCxe8->e6~Y#Tn-r>iB9|;X~(uL-VY8YJ7STc1MfSVs2Gi0xTGe=Tkk;DK)(YmT#0tDZ"
    "ore`{Eb|K)*xQKHZ~l#a5;a6HUNUP=L*b~QMq}P0!A~Jl)b0)(GFI)X6P!j&wydnwsVh`y~8J9%v=_bh7>PQNrE#OeZ%gfy8AerF"
    "gXbmhk{9wCjhsZa2R4bT1_+vA55v|2`GO_5vg+s63!7%5(1QCODn%(<gwt5wqsa3y`U|Z2HgKS_=CHEMsSj?k%pZFM$<q5gaRXFR"
    "bRUe!DSANQFp}fh{J6r?6KTBh3$ups68cY?zUYKGLfprC^=`}o+@;r-"
    "oZVb+@psk#LFA$j72Q&)C7zQ=VjuSKs4Uj&%~gSM9oyh7D_YDj)Su48EG#)kgY4N-wpK`HdLS=1W8P?3SOYP<{FW4maYI$%8<@-"
    "mtgE734%f_(D-1pKz}CB>L#rDIdrz{HB@Dyd?aPyE^rxW@)!6-2jPJEHy5K}UPaMCS;P_D()*g)jt1Su-"
    "z0?UH~|ulA!_nwGqZq61NMQ)G&SJ3!Xd<=mxRc+$V_<b58L^3nGv&ShS2031~;=?dm*J+7oQ$-%d~tmU*?a4-_2HEVIiE&6s`-"
    "SU2X_$B^bcexa6#8XJgR;A-E3);G7~w>QH1HGl&E~OEJz7n=Zp`D04#EokSAe_sg6H8j8J{$-"
    "YV?zpzh%HubofIJ00mFoeL0+XB2SFgrF8AT!UCOi!KK9eZ1nvPd=HGD3joJ;G5)VK;3Idt2kW79{}wotp~D4o>kb2c-"
    "ID$sLYhnUYX@(lMexyVBmPTLwGPY~cKo01QIW$tl;7YL*G~G6TMuKAQ73c(<QT?0u-"
    "Di`Z<oI2&kj^J@Uvk@ufyngvsX!x8we(d{hh7>LQCd1|yXXa3-kv#k5JD_V79m?`ENP4uWT4Q7gh-qCgh%(eiV4CNMyNs()D3?u$"
    "5`mDXDEQ{CxJ#2=fRT}7m;8Uz#(~H7&3DyX<!f+f$O`)MTjxejmOsbkcf;f-{X#ak6-"
    "Op_)*dNF&PNqnkQ45Vr{jJ#J{DTnM(l6TDW0StYR!XXc5cKm96B5*XWvW!{LPGQ|vpjlS51$nRX?oz0#_+&~5rQzRLum-"
    "K(W|UQ&4`Df&5$tTtY#up2j~pxkxb|LJgdWi4k&&vT$|y$Q*=+PoKrnw0|7!1o*u2`Ob}usH~`*=BT2uw5f8t#6oJz=`^?uOS=~h"
    "`AU^;qH2ERc3Dq`pI>=vJHKfRzX;)CVeChFrHU+Y4BUAdy(rek#Wx{g0IpXfwMAbg$WF;>`=b~G|lttX$62OHqWHk)r@f`os>??$"
    "GIt5RPl>lKd5n7%T64VOtJKDkPyD7dSRY!6|sm4~wej5yHIKQwlbKp~`+&k!uJ44}WNxge*KY^oXGJ%*d;@0M4+6(XsfBIPn1H(O"
    "=R`fXuU<TgP5C-"
    "T3quL#e@bAZI#SfoxYqmV}QS26<Y3`>+G$LO>zXBne?m)n2$iO$C5JgHLwG1iKwm?aO1Ud2t_wUSe1S$wlAcps!9iP+L0rTnxt&b"
    "QR`7(V?@Gg2@an=)^FerrUjdVH7wPjf<W{x&He#uEEPUBWX-"
    "4P@$U}d)CNA@?0tn;v5va|>ZanWF~<gf=0+=4hCS*RWdgE4=0WtUaPe>({V;}g@kAX-"
    "`1VI%>#nq7)9RC+j!;80VTFmlIq=FbDFu!;BR%<L0G-"
    "nK;=DSR<1kWJA6#jc=*ehLFXtM!<x(#x`t0tKXRv!s7@836mh=6oU4PsR;4BP0y=aDG9%1!_Qyg?RWRJPZO(HL%#mqb8V;K=X&R>"
    "f%}SVKyFbY<t>b$Q8PUf!!U`Tl5&}gU_hWdF7b2aG64}4_&N!9#B;AJOWIsiF#oMNWX*3x5VBMm#`N;Mz&u<!Nblj804kBK6E&KV"
    "9-6g<D|+gb#!%@_Jww2MCR1(##g4cp!|MCV-~NRzJo_b;51fl@!OqNdTe;eRZo>ogsm5@%{RqU@a-"
    "HPDY<G;DcOMMer#)m10zMv<gy%sY@QJ88cq#7imI(o?SUW*sQCv4(RKk4C9B%$A_|9!QGV95WR3%e1*ZH4qy5g$yNsH^jHDWzH|?"
    "c`OTsiCFSjQI8ix-PCQ?X`hsjKVbQ(@v7`=44nEX;21Un=;Y|UA}2`kf+M{&1J4K+Kydp3&e|6~p=yEtM8aX9^|YP$C4qZlZPD)7"
    "?rIm7u1+oFa@pi;8kSgu~$`Yv2yp(W!SI0vH@D^_xJ2H)wfF?=@<K1<gtM?Y?G)HEwv-R@m-"
    "J|{UByq*3zbfjnUX7t5r@+@B5E_(TXA<X6}f8QVWXVb$aNUVEW;E+56^l=0M%FOiChcV>6-"
    "O_DxVNfvaW&pD#J>xL^_YGo%oH_Ny;ncq23x}i<9bnNx(g?VgS^}_8x^d7MMp|dG38}EK@M-B$dQcKGQ_)CiDr%oyF-Yq`Y2w^PX"
    "7_~){yJ56+k4k_20Gh6KxBKCqTWlEB$Yq?Iud^{#B8u(VYe{=2TaYpBso>!rTC4<0YC9LK+a>z%PKMgEvsqXWlp`Z(-"
    "3(o3Gl3#e35Hmuzo0zvy<U*koZAulh{tpCv=fWH-"
    "D8N0QQxYLNr&(2bap00*CCRjZ*xx+Hu8&N0DTR&m4bvi`S_{R_oZDlAf#;9uGy9@7ESL#70bM{xbg#K+cpA`pLt%O3n$yrJV*33v"
    "!i4f+Mc&u=$TB@wLacIYp@B0|&u)nqAQ?K?w>Qc`+LRmWbZxOIM+dfe|zZK*-c)+p^g-yJl&3br?+}-"
    "Z}U<C~yP=9Dv3t9zA=C7Ay@`y1DT{W{lzSJcOKS3Ay^>kT!77%*=pkBim&qUOTycAq}xIanPb4nA~e@{Befn7h6p=ps!BwgWnl)C"
    "Eg-EO~A7|YJe#^b?~-h&y9?(Hl{iBm=OZkaI0-^gKm5OEcX^iHN750g-"
    "K<iV1qS%QOdTQbLz+e!s@BB`87nd=|L7PlQv5#ri)oBu=Fu#*(4k88>OxV8x)d4zor%~7pj%*?3bXB7(HpLji@K-"
    ">&uBTra(b=w;g^&6BetB57L#F?kY}WiK{(lH8TSn#0F8!P3bHGgk@<HeTc89!4JrQU&mQE6He4!7^foit9BZFFo@ZURxERVlG?#j"
    "wPtzz>REtMv%L1S>15na8|G(&ZbCr~>=RNmliF>Z5<jEO$O&&=deFz79SC6G6S*?}dscxb)4Z2XCIL3_5!WP!`5P_DK}DQ~DVL$u"
    "|8;(L@{EI>vpt@?hiz4~gs?><`K3dZ8Ja25ew9p8OcaqZqc-`>QFbxSWg4)hk56wYvH%=)KXbe>BQL6w^>a=VbB-"
    "E>c$>G{JLUUvW<SbL>UbNPN4)LNt%YX`s#2<8m^BP`PWF7<Ik+^55|NF(EE%B?!LdgAac1#Xbz$>2qD9(=<WF?u*DF(yZ+SK(+pm"
    "SBlh!rdXI?9Q*3s9)IUlJoqY850e6DT@A!GonG@Tw<3oGf-"
    "=*&B`$>X=fA9fTjfY~D^2K=w`e_V*{sP7q2lYMMfA8y#tP6Efo%>`kDy%TPvkq7lVx$ed4Wc>JUo(K0~k(sGC&ifogw2|P7q)(W&"
    ";=?vP-"
    "*3`s*gK=GXhG#9DK$fYkVR!$Vx{isZ?0iu(gR}Uk28P=no*l@ffo(7IWuRhqMI;flNR_&r)9>)BY?u6e@CeUE_B@sxcx`}u-"
    "MR25|&;CNqVQ@-"
    "R8Df5_AX)VEd*trgsLN^AF`5Ji(Vc>@z_ZZtDQx`ca2%&I#+8&go5eH8|HyEkggEn*>%T1KgN<KpQIu16&VM22ZB=tf>n~9Nm%u{"
    "vU}};iQO(5rm}7lOQy)Ht=Jt(ePv`i)BTd>I>auSo*R2(T?<NnqCh$ctBnFxRt;K7TWcUb#%-LIx#kha|GQ&`cz}PK`#S*YE-iw)"
    ")>_yzE*e|WXH`XD|4xl@fAtmd<1&I_`&2QaX4}07?4PIDJB`r-"
    "Qd<w>6c;mZ;K{r#Aa%1orc!L=Lv1AChS;*OeeI4o$xQ|JlJA38(feFcQjXvj2yC6>LrghYgw*&+9V#wOrl&h>%xw#oDsC36M&{%@"
    "vI-u*-"
    "3LjG!{fu+nk?rmd4>iOV&8@&VtX{9{&*U&aZ$3oAf{=#b@r;C>%VM0DaxvYd;PuoMKQh=BWn43Ajrdi&yxmN%pw&EL(Ha2(NcUN6"
    "}{NV5&PuCtKWiWA;gah$f-iIf3w<9x~lI;T(2}@7aPTJ)N%ZQI}-"
    "Z&wRWKR~HK6PF5mF3}e|aZ_w6!WmygMNmm7&X3yVA{x|bge{_gL=PI07MmENvkxp2f)IU*?lfpycpZHDYjYC|6^bXhW=l%%=1A!J"
    "L2yRBj^E#eFMX^B_y<!DV*kfT^`H>?peYU`cC4S3x4zoQJHQZUzSiTpA9X{Vrg(`aNQ2z<iUqRR)zXu0Ds_B$=gOzmSB01Qd7C(n"
    "s!{oHc?zEklOO1INM2Q%%>Dj0Md9UFOhUA^fd<Ei6NlErhR_E4rBG!z&7;Y-"
    "hk}@=V$tzv=HtH!{r|}B(58EjHKs<BsA>r_d!_@wIaE6oZSKjIfEt*>?(?;5POs?Z|2IdUUGE4F^Z02<c<prhL*p32#>OGB$IoNG"
    "W{*x4vQd*p2bJ;w~;ddV!7h;^-a#eTRh64vpl%oMUz`3`LqSZGmTu#yd(hh@zeR4u2R6vy2WVk(TYKhl`_qlSS5u7;-"
    "ze$vPk}VvSugz6PcvAROu#4_9r&noAq<F_py0lB)53yi|0Vd|ApNyP<40FvAT1Af&Ccvb7{e_7N(+0wF2QnB!Bb$rlnU4r<Tk}b>"
    "mG#?oAFETxnF#Q+5N1H86@)r3vg9FaBe~q3zgof3^HP`t@OcJ!vBDs9LM-"
    "t}`a+M`rr2`Q;}|93woQWl_sW5(AyI+EM~}GNtR_Ok?S?iE$gfmldLJ+sAU3Q5ZUK8q{Gi~bxN~}TD*8B5_#8|FhX5b1W!lx3H1|"
    "s$G(3c7em@>Sw=@72_bJl}0_rrTqg#MRo~;wOx!7p8nmGY^0W=LWAt1C~v5~o>z`;0|REwFD-"
    "^1;~j^{7wQv;I#dxW&ea(PYJbnS=n<(!A-v=w0*44LTTVSwZY61l{v0Rup;3wDfIXr-Ix!Ceb3rzaHV@ET<0<U6E3S%WT`0ak7AR"
    "Q+I0`%YL40i*3?^nmRkm$98>;P*5Mx{Jo&Wct@kAtx1NI&e?FY+C=_1l$T_rgwBx1SxkS%Vo8YVWfn;lmry{$kS-"
    "}zjX8GN#dcw!kOi+fhfXbbgWJ*ju(>&5F34=&Wlvf^S<=2Kzx71HKy*M8itmi`f3^`fB`f<R!#cD6TCYIw5j7rZXzRP;H_UM{bf`"
    "|uNHY8_BTmKgK&<Y78eBPuqu;fuY9^GEd~gqO`|0p0@9h*N~}`YJs&Q)*)lUNo=yjc{;xW)a@o^U&r1f^{S2!g*JXuA;ZIF#j3(*"
    "G34zGlJUj9G%@JFkbZWs9nNXsBOk=viVk}!j$}6fAkBi$iUa@;T96QMNdKG1k17ow<YGwzt1xbTX&abQ?z@8z8-"
    "6ilto)(O2wQO$xJYhXJ3G$5Pw72}ZR*XpnHLop+5)L}+e8Z?(nu}xQZFy^rYTw85!xQ@Mq<bMqN0iGYy|gbR0Ae`sre_z%j)Dfe-"
    "8J*%_?0rt&y(OTU4&lUosOH}v%w$OdnQYCm25feR_RRCZF;VVeal~tuE6yIfSw0h3V*aYSpGr@Ejt*tBSR)lx7@pUNO<vcVz>`o!"
    "!RIMw-FFV6dTx28B;(FTc9on7$(}3D_<FB4B#KsmsYOUAT9<QG}lXXIzke~xCY$TM$S>s!R(s}y?{A5Iqy2|O!KAMpMacUG8GMaX"
    "0O~+F8XK`oy1Vt)mX{n^4lh>jCrY(0lH02#x3DD=!Ll@AMAyG6S%WA9<4x9u~hQmGsPgW&!=AXH3l<zJSXa_i3`xh$*W;5{y~GZG"
    "=%xe-E(bzGq{h-"
    "AZ$#&oHyygkcej~flnF5Q9i+7N?U;B%sh7*2K7)TR~VI%X~9^&*_vRot|fB#bY!~$5XnkId>wZ>x&W|DJE|IF$j1Ee+$j95og1VJ"
    "Co@z)bor(n#SSxy7FCOJR$mpqjB~q|>|i01NjrRo1B5Z{)tCs}@P$sBEaeJw504g)&yJcH3+e(r66KVe*ja6vVFw_I`P|CME+(_k"
    "PR0M^=NIsJ8ZfZn%9diE)E%=tV4vbE<4VuCD$d}OF5SkzraM?RGL08PGDTg48$oJ-"
    ";Pqb*j>089|CT=B@N^Zh<C^kk(YWX0Z0P&3X|R-"
    "s0*&tfORF2a#Tl;69Gu*7b9g2l_i7J?R7JxI_Vf%a;7f<hKn2^>8cnXTDOB=wz0Avv4To*VfxR}J_~q;Lz)J>&z+lgP<a61TrFlP"
    "@`|yE?S*`y65?L}FPJS3%$R7&{LErZD^mJN9&Su`$=RtkRcJi322UPYIiO-{!CQaSLz>*~OQX+RFtMF7J$AbB1rv-"
    "}5)^Y+l*;FSjf{u|KN9Hm=Q&+pg!JpH#57#oh_wX;O6zzIg^9D02<8b!2z1Qd)W{(<<o130*m^xEajv4PtIn#egbPTy<F(X4L0f@"
    "QcQ0r(j<me20?-"
    ")+ojQQy$DrepdjRQB;a&3Zok@4DXA*xrnDT~JLS97XRSD{S8d=rUVv6S$+8=nd2CRfFP6$sOw*}P&dMa%#S*H||*f6{lj{xePKd)"
    "Yi+(sZf}2)cQkU=(V;FqQ|aST_JY-9#AXKL-edV8DJmWXDN%M5Mm$2kyDvl^28B(ZI5JIh)+E0-"
    "Rsq^OtH33T=K}8!YMlq17I#7gOo8V8tvC8-%5!bykf(SQj*qA(>;I7crF=cbaJ@C%nEtd&8&@BUhn)7f5D@h|(|-"
    "A!$sh3F?n8>9;xE(T-"
    "`JWy=)T9eo8l2ghi3%NbDEWg!=S#P62A^7Ja1*i1gjWh9SzF8h(=dZ*MbZ3LFp>4@@j=mkj*MAmH`LS&{k<uoODdNf*o$+6<nmQA"
    "blI^I;6Xp^QV+13>tGu1)Zzk;K9RiAO%W(%}P06ejboCJ%TXj33+!_yHS90y)p5~RE9F+H@`b7mpLbi2fmkQe3%UxZdOx}P#y@c5"
    "SF1B}V0HOyvtW<8`N+6K8LkJ+`bu!%vT&C|ZIi#H=Y%HRbX6`HDWmF`z_lNZ!sEM`ebf^72{h_Z2hFR@W5{V;H{bS_fUZkZpn8c@"
    "G-j<uW{cndp>HESjoobDCeGsrhJAOKJRUjgXvIi$jKzag4DYuw|!p?zw@Qp9ROjf9inWk44-"
    "vnkMSTv{>Rz=1RTi1blFX|zyVa4L==E?j)@Z_DJ;w)xid6Weki$_}+EI1p#$t;ydQD7f48tLEWXOfwcxM<)=6K?;lvrH(s&7R;Js"
    "%7kZ2e#ro~LEUFJB_{2x8pks6jzZN!*vpX+7V&uBdu~&7lCT|s>gU#8mN<`5rsnX@oJJ8WaN26${QaKMEuJP6o0sd2D(YFJJWXPc"
    "p$y#v+>wLRax2GOn~<(+E%x0RbP$PS@xorrNgT+Qk-"
    ";D9o^u>0O>;ZOq69i+&u7sSlHSi2PXSjhQ<A6*bgGoOF1=!C4zNhT020`wV+Aw=<G=VZF|2PG%I@XtfzJp_7#ecelMc_dX*T%EZj"
    "jC2E(3XQ`&D*G%`jY{#rWl1EXwML1YBa3$*e1PnDBiFQX6?3{Uo1OH<^+&*1Ki@F&H|2h)XAq0!)#tsf47|)QjD^s#*rNr^77kqW"
    "&nGa)wR>LkjoP*iqO*csdOsJ9XiYXLalim>Ccfd^{gb6J%wTGbZfyV_v}=#k04v^P*}h{Ws<n^T*IFn-TN-"
    "u@duS#9Y76BMD)IBIZw{dpdfg?8_aie7?<89rg)tJsGuBh482lGL52gAOyg|0#W@;b4rM1%OZI?u4zL}Z7^f}&=*wMWxQ{D7UIh$"
    "H&xN_04AEvYo?R=vUb#-"
    "*bLA5@nEq=Hvs!{2*ZvMS^+j=ttN}G=K*_{X+&Stdf1aQRTujOmTlVbVJZ!x=?g1&fo?q)Sjq)dt8@jG8hH{uqMd@4F9jutUQglZ"
    "NNj;|7%e@rSrGj@c)7>(rIY>y()^s})|?|q%-w=}yl6%_eXQwzOllqrFoBtk3gi4zWG66KBHKBq&P438If5`0@qo^npty-ia1wSX"
    "qUaDCPHG-"
    "5l#2nV0}D9_7^9|d@fH_Fc0*tCQ?GdfB22~g^YHQHGQ#bOetUwBRBOPijlEt(9NV1J(YMiS(mhy$i2IqEmT<gHY=ETE^Y`5B_}Zz"
    "=g8(vTjmw+HlM&z|yN#B)Da_PV%s#082Mq8>=)`IgJCB@9ZDR&(Q=xc8*SMjPyp~L?6kuTtKOqaiLrI+*_7r5a_`I*wk$Q*;=>}E"
    "QN(TfyTj!3!m1DJ>Ze?mFZCf(PYOK+sU5NKc`A_!XKA7C}y9n>)DUNZR+bC$`UadpsRt~5}XljEDG3jk%(C&t1o+|*3(2W?lH_Pa"
    "YB|uDK1DAILqDHf!@Cdy{;yBi+5EZ_H9LXT)F#?w5aAX|H@Xo@twaR!b8Emv=^+lp~)|fraC~U_XeySuhWcBvE+&M-"
    "CvjtFjstVw*OyYqCNe*cEjtsCIQp?6D_M9ri5i}sWyJP83Ai6X=Na&!~BpQt8wP#*lM~~23Q7AJ(?K5*auIKT*F+=wt%gD5rX}tI"
    "zQPY+Lg0cZg4jq7+gMcu^(}o&5poU7gwL(9W%D3J_O!;Ke1b%5fhmql621Fi%17hMDT4h@QKsyn0ana(d?1yrzY{YtPT~n$?&=XN"
    "pqn=@M3@}Ndv@a|fS!=uae8+@Vu;?9%q2=jC2HiF}dX1aHqZ}mzu<oy?+ycW3Zgb`(>=D$Qq{lPe{df(>W&NQiQ{y8z`k=ddQ5Gr"
    "1<F7C-c@}1;#Ek@<1i`^D-"
    "y?k~<TW6FEe|>!gPePGo){4o=H4Fei1ErEQK>Z8tyGdPWIcsclE{*ux=u@&fG0*NIw0z0Bh|w_C_&&x?8K6zNd5tpJyjR5gO(GN8"
    "9M@cYd1Bs+Yu&Ne#dPwT1F`~HT{cdjBDg?CPWN%rM-"
    "$sBi)FZytPb2hCi3_hB>{kQvg0*dMh<5tLG}x^Akp4+n(@td|w4H8^dMVvWM6e<V_l=(nX<v6^Rbadz+xoXrpoyhCm7b!Zw2GqQh"
    "h!#^+LfAWaXyr4c*%#GKB6hDuLJLnm=Mmz4N!!FoM^OhU8Lh^Yfu()McG)A7Jlf$z?Kx($PSnAN^`KfG##_xP2+QnkZlMi~PGCN{"
    "p`#W4#PWG<UkR)p)UW)cUmNA{(s=Rcr{p(`GOe@`LyK<$_iW744QDLk#K<@{9K*WsQqR3Hj%drep@4j}l-"
    "gBZjQ<R7*3&qe|Ull_)&|9Ns6frL?p(MzVRI1Fi}({;(A<}l|dFPuG`A92cI3QZfrs9S7m&xJqB1Hq!0aVMMJ6atZz$;U;T+szta"
    "AZ1O!^+adtemtDYi5?jQOkCS>WqtO{!Zl?WK4xG9Mq8ojIAc!I2!lLq(5~ETRax7u<04xooy(Hg1&ODO>>Q&FoCQPvV3Zb57w=<K"
    "RWw!MNd^Wo4Z|b7z$@CnXm@|AVay<7Lg;6S2uZ;(qbUwH)2-"
    "pXA~9|WpB$2<7#g3bptc6SESRg@WSTQX!lpmAxvzmZG4>Oh^6u%k$ePFDewpK(wmXeXV0fptq`rMjRrH($Qp3^8vjLnDA54I#>UV"
    "QSG9>%iw6O-"
    "+4cimq*c>8?%1)r%^4wdxxRa)Wa?snoaqv@Hpk~3VD4kZQcA0)++n()9v{5yM%rY6(8hdEg<hqzbk?8$6UMLpffb<qV=%$ULx<k_"
    "b#j1gxI6Sxg*_JmqDlB3fDRc|8QrE*ItU=#|5hdxD3&mtx))8{C$l+-"
    "PK<c&Ca?S_;=*f8#JB`YN@}kCv<illD!j3=2@QBM&uaf$5ybb$O>{NY2!oVc#nIRTBCyyktC$%SSgb%@t{yc^u_J52-jVS?(_-"
    "xT*#9??Zh}CeaM$2B~c5KCIbGo0G7=AGvXOz6RrqN7@79Qt4LS5!+pRPIdBtWWH_?xHJ(0XrWjm5+)#78&^Ft8<j704@M+N8T7$>"
    "Bibgl8am`<L_;mgzM^RJTM)TqIEjt%DL`NxO1#4Gd{3D#~O4%-p{0pb$1q%Aa})mt8+G-pc0$k-"
    "?H;4&gLNAh!vdCx{C<z6qiSxWHKLH=zP$`9XE^>um?m48Kd-qqMzm8jbNuynsLa);OLSsEN*MrVSmfbY2Ze>L^4VlX1)yP%V+i-"
    "tyK&k;P0!)D{79#xOaW;P+-jkTVg-"
    "#Dg$C?wkI${dbPXYvmjO#(;)Oi=L}<=`gc%Hy@%six3Xyg>S>|xhtibSacxWw)LxrbpwClLV$<o&At<GvdQmwxQ|!mKW2lUjj^EO"
    "l!yBAYy)nuT<b>oAmN}p@&IH`Bsu+<Vd5EGK-fm~+Z;-2j0J@282kW$e6rI4rIfs3dXkF21oM4Z<cyTbS;)-"
    "zz}yi#lI2jA$hVC(NgCcCHb$1;c!IN|Yr$He1P4MC)QXVOPKqCn8yF9%{9t%JFsf5KK&q^Yky~y1cf+4C6O+!QHIBd!t=tls3&%E"
    "Rph()FLkwWB58e4!Ti3&DAgg0Mu**Dyjl{X3sPWQAfmxILkBR8Lh@1}Db316yQtxe_nVy?i;%7V&?BZ{ohhb_&$#ld0__EeDKy4e"
    ">XV>+zJ;fF1;-m1AeH+xP;lUitstJ@2l$J9SaH2-"
    "E<NLfU0GVa6MGEGg=!p>*;hO=n?V(K1P6aNL7*vN?IvRYz@n=`zz(z$8j?}ag;A1`8mF<Y~*cRd`fR-awHQh=OT&po3O8zwV40Lp"
    "m6a6N49Jf5+9l;9BdV~eRx?ISaS|*bgyRoIY`<t&CAQ;&XhMo*;j_SewhG_1}NXwHZT-Z(n6V=eO5#RA#_MQa^9{&`9-"
    "8Wh>nsJPb*CPx1iqMC<78+a2)%6f^a1O~RS{?fRZQZg@ache*5}2z<;hsuL&A_e99O-y$>`iNMoD^o-"
    "u*ba3ox{bMFv6Pctgkkc=ETSg+cxk((MZIGKuY(S?dsv-NkXHb;b3-"
    "yE8d$~rH~dE5Z6AxLle_8_V3NQz;G)V%rtW~y|}5723>`Yq3ZpyrD4GhPltMi&wVC7IniX?&MA9v#*cw!-"
    "|`J>)_8jgBGquOX(r?hDB+p7)6!&*axJWND2ZMf!m<H*9Lu`T(n~Ls9%+xv;`Y#udJYwgw~k!!?0^Aju*c9o>h^9kGrBeLhylnsi"
    "cHR^teyIsg~tZ*xWu?}D1Kl&2dm7m|0K_PHuEEOEpQN~3JzFooKO*=v_1?KKv0s413oM>9>~)|lMh>p0)Tmrpda43*Mp$KxF!h#J"
    "S9dKY_kk|p&K|Mfr6pLI+6Vv0XBIR|Bo<a=Ji=ah$b_674|UF#@XbiD`a=uy?FXsJcx)vaEjokko<mWI2k@OJV-mnubLEQy!yDC$"
    "uO%TYcdl6jS6nZPL*l@5>1D}(or*r5GW1%>a-%i?H)QsiqJb#u*xtN^5K{j)zYYyhFJ3)qNE#mh6);;u&vA`&VggR(1Wq-"
    "O+Tysm`Vzu<@Q&1mh`#Yg-wvfE{8X#Jbpnf7(YC30Js0dMzI{K;i1s5(Mp9VnHdcQ+Bx7-"
    "zMpP$J31X~nuG$#>vp&nE3Ykkc9QgEQ!!^|IbY;BS(vs32+RVGgQuf*TCm;=cpcMsdr%1O&scFAbcYj9ZWD8zG53T}2G6C`CLg>)"
    "ozpM@GjD%!G*S{WHSgwZM|+;4KR~aUbe-!dk=;4+2-"
    ">RAA=_kPavB4#mg!XX5qmu~3LZ!YY(fA9!H*P&J^U~agcaod<?ITP9`i|d3P`L%^kBMW8{8S$nI|A9YBRelR*;i&mPE4-"
    "`zAx1)O5Hw!xN?GOu%$NPvoA}n^=+HH%0}7CoIiQrGINhI*==1+f=6)%2<998y+v*p{h2VQ3B$VxqRhtGwm~HH<Q~di(5ANW?4eu"
    "UMV3!<cUe{^Z?IlzCq>Qm@fn3=dcC}jcW`8OfyI@Om4%($-"
    "^d+NA6D?Q1QTBQ}PO_hOsT$WN0K_7S|YAUbnBy29%nok;W49Px<SxBRxSF+I&ts>^{*r15CE4?xY>Guo^t=*T#f>Ofw9+LY~zaq="
    "A^((|+a*L}*TE%q^|`HSO$Cg468%`KD~!vC&SpCwMHjZ6Z1RZW7`$u7Y7V>2!J6-"
    "Y8xT+$Ok|8&Xgt024Vq3b;;s_nRR^vlN~QQ&6hGiH$lvK9(_HV)|0yqhYTms-c_+kN6t{)LnLYfWXKb5H0-"
    "S$&3WXX^}yyI_jDG0M3f1>z9+V7)A$d3?}=^iwo)wn4nX+?swij>Jw%t&XMUcyX@|a&!RmGLIcCR*khe2GaaG|LSX=@xZ4Kbmjft"
    ">blhXd#`kM`QGQL6fWde8A>-"
    "qq5MVG{j`7o;2y|F6YX~N!%Zkm=+W;m4z4RCj@;6}kKmTR@9Vxf<p0@2NpG=xL_ga09Z^w^Q?~?c?n>y$BU~|oJjiR-fW(@iR?qZ"
    "Xzv_ax*z+jU}0(O0`dD%?xW~-kX!a;uklPm_#XC4zqpfEBiT+4AxPo9Lx!py&M#f^y-"
    "Y?PBAKa3naU!a=zbhd$nM3;(LtfwaULbWz-aR7vJ;I3!22C;^LZ`l=m*5SKNTDUDtG|~h#t8LY&S5__5a|p8O-"
    "Hvw>zR;XLcUFo`y%rshl{hEh$hJ7*cuGVf?<^<gkWZ7r%O;!$<Kfw~Jc6_2VPvKeVR#>X75?-"
    "jb#2~a0n_0mpM`^1BzY(s+l)tJ9`P*bv*SgDe`No|wU#F5W+)_KJO&4Cd*K1h%oy`F4Y6K3EMV@O%`ptiK08I8n@I{Y`RKTE>QM{"
    "%uL%&eBq@9DZ=xGg6cp0;E8(Dt%rN{1J>*QG@kncAfq>Wvd_fH3b%W+3F)L#Ln~_k}w(=tl&X`4fG&XWPuFY<-"
    "6S5b0B0|{xTqXn}4GI+iu;_VdyAwJiHYeKONwmRmJ#iEwXIP=_l;=ZfBA%O<r0-"
    "0*09{~;ahuMhZ^%W9hz$kf>ody*?41~On6QwIcQgC1ryTrjc1ZSVq$6qYoSOu2adT#b_%*(omJIk25bmzU;Irm(HZ)DJDiiHA^2i"
    "BHW~VPpu!+_U%hq#hUpBQrY@%qQ__>yTVKGs`Gf}@+0MtI_@SB-13aa8DI2LX)NNpx83L)0jf(;jceA#Ku4~+Tz&==Az*iAG+i-"
    "BPe;BJSsUc+?olJ}@~?jk4On9kBXz6#&RWMji{gi**M%$^sbxMfp3z4JU?*vH4CFOcfyyhpT|Zn-dQ`76l{AMhC^KPjCw?Ubpi3x"
    "-amn+8%n4D=cDsiDE7xxh$Nxy4a~y>*uH!7}Y47eYEkTt)&>zer!r=6Sr)`9p-=UA6=M_nQ;fY}oQVl)5TRGh8@(*saM0hjK&{j-"
    "0GV9~`oyO%q7SzI1gb+(MqgETtI5jLB|vL|2+Cg;mXVA_(f!YqB5wzm5dXB$cEF0y=*tK0Q-91Gb*j-N<U*_B@vu-"
    "pKK7aSH9UIL&a5!ej&@Ao)QewTu8V#}*@l4AvZG9tLoKC@RKl6gZ%EY0iP!JLx8IVYeD1YeogCR$^Lt4k7qLn1+AGNq-Itfb5}$q"
    "^!&u9{7y?%DZJ+Rkje0P&HU}TjL+hE#j1X;W%cH5A7qw{2FcCTmfRPg56-"
    "7rEQ@tKti}ZrYC_ko15DJjv?3)B^ZtQKpn|#vuW&jjph-MKZG1)A3*;Aqse=)KKK7i5ycDx9lWE_sjeDxjyZ-OEU&WdLKg}qjhrZ"
    "Z0?l$s%o;j2*}2g!Nic%|mZcDdf)ZXx7jeoekrG*s9e;UEIpB)sP{&2@xcJKLvrTca*rh@ySTOjgd%QUs$$VtaSGXY~yS(~d(%G|"
    "Zx}J`=?woqiC6=-"
    "=l%$8z((??iRTHHdJ3!A0mg>r0G{#HE2HLjCPp;$jTvmei{)0kGFTUIJ;*~<kKI+WnW3+7QxR`#nEV=B~^^}WlcV%yRfI+bAxZ><"
    "cz~gYKQpm-xlORvMPhyGDaJxNl)`gUlip8u4AC=KFvg8>*aEkA;6r!EMLcr^Elw%k|ZPyQ^P_M?MKKDF*tQ7M!QbeFS-"
    "|p&V_+<NJq>fPRe+s~6doA7nO}--M3awsDg;9iLNU-"
    "0oaVrQ_nYe}qzB#eNdj2=61@;?b2u7L<(zaIcbGwzd808&uIaFH>8g|Gyy1hOa@llh~nlU;0EzRuAG*-BteoO#rIURzQl-pXy{rh"
    "h^PQSKE#C#wD4|>KJe*=+0TTX^I8b9RwsH?tFD3f<6B?v<*-cR97+&XxC-"
    "x?f4l1CmElPGe~Ix`iaMPNb}8^fArgoro_q<W8mRva-8NS@jx8gR$hjlJ|*)T(q>nJI{w#r?xk7VZobpl-cZMkqblXTLlqqdA8gC"
    "fGGodH3|u1>JJmSP7=Zt1jWTf1i_Iz=A*_JUP(*pC~!&03fSD5X_vp8p&&T)Mh=Xlrmj<uH~Rw++B^-V-"
    "p<!I65Ko=lY0j0{1JpP9voh42=i7TDHP0b}He}Td)6G)Ao?TOBFZE_;Z?lU&z4-"
    "a{1UM6nG8~68~EM=v24o<vmY=D&YeX*$1M5dVMCPgAG8YjFLTuJ!2cb;|-"
    "0kGBC93wuwEc%F@MvbB<lYMAqLlR7Bv=jmam<*x;JgGZ0$^d_(?%BpvLYv^mB^c%taEnm9+cGFqF7rx;VL!sb3Za8MQK=BT#P<;_"
    "-900w;}N0Ol+xVO_jUB5}fRsq<az1-l~Nl>SnZ4`D|uK4JJXa<N`YED+Dr`MkKn+XPE1qbbBpTHWex-"
    "CWT9-$S_hVc6EIb)34O+wCfd+Y!?ol*WXmewyjz>ex$6J$-jVkBL9Vv;~G<PX7AQN?M3_K-FAY~zAtG-"
    "C>b^PY#pA&My_RuXsq{Hc-9f>7dEwyQq_oPrt(0(E+nFw&EhZ_t5Xc{bW0e%e%HRi^Fc13~=ZQBg*?0PZ-*91o(-"
    "OgTv6!$jHQY4hcnJlV9g-K?f*xbn8c8!};PJiYPAK>#*md<c33)~$VAp)_U-"
    "&Kx{@e<=IAUwQk}&cHg;6A{b9eVIyQ!yBeV7Kq07(iPD*gYuen%G#cj@!3{30|b@Gh(QRjJ+jUx4>iHyrR5%j2L{WH&VTP|W;d`$"
    "qx106)ErO9Zg~Yg#FV__&T_H0B`F@&+hy2wH-{EG-"
    "{5y*DK)7l8uatx&o2CJ*(R;xrzdF<LGWqFsgTWv20orK+(_yokOmE7YRe+bM3?7zJ`0V`ssa&GbN)kpo9&l?qsym-"
    "$(fm^pGYHc4RKfcr79N6JuACm$Cs@~OoXWS@KUS6L?%<=Eda{<rK!%dDwK0-"
    "{h%Xnf)nhn;_;a+r&=A&_bg#O=<~4Cq1%yPSr&EENN8>c4<8W?cH0=5*mEq(dBjcV@}>*;EZ88rF>H^#Y9zT2x)?Fbk!A#fLtFS!"
    "3J@4?%@+8~QEr~0GE0KGJkkor8Akj$50(K9u-Yt$hFY5^E}W+6LwA>U5(6*~y#af-"
    "8Z({qfc3yTp{|v73p;7LL|CVfwCiSSf!)qH(Sw0Y8$1BSmv;MDO+JekuO|mY7lux`$+iaHUUE7(=?8Ye*$MSzE03DN$$gGACmjp_"
    "ubCpMJX@JedTcsNCYDWH>T4bjRBBkXA(X<!&Yx)}`b}#=^zd+*P3tPJ!;X0Y$0#mW<tR5ToFE;FJ@uJz$Xs45!@fMiKigE$2?lmf"
    "N;)MrhUjp23v40K;79={c&5E05$oEO?>8UKrVtaL)k?>BJYKZdl8j!x5>&EKz3C*3oPrD1N{^FbAsIk+QM2_2`Ed4r?3T-"
    "GG!1(ePYK16{eL}IBB<V*2ln<$52V!me%9e#k5<_=m<gnkBz1)&a#jt{h!;-"
    "UxbDU!(Z~fLW?1hiZxYUAgeLfw2JlX67NI`lKFi@g>KYuYP!{Z(HN!C0P9SljT^M4#Sts=9YEp$pVCnu~Ln00C0Fpt<h3sDtX#fR"
    "3uijD8dR91cH<0g@wXn%E+zNS5J_*{axKxshQ_01A+ilCwI586oLNWxS2+4{KY;scA39nO95=xUe;oPvZN{&-L<-"
    "x>h#@!Q|5%9%Uw0SUN<XTYyKKHO`M&Q+u%zk=V&;dLldyn5lVVkz%hSg?xG_k}0&1m)Y<Waro;USWVCHd?jN=4BexwEGX+iJP@0_"
    "Zf7h=JmBqVwL?N|g}=5b|-"
    "XE{YQ;B6y0<9WsL=(j%)!OLUAIq`(OE0s)o4QbHupCg;%oETb$vIfmjiB^hptWmlm|SPwJUaevt_foOyA4g!AoNkJ{5?ot^<;R3#"
    "rvks_j7>A7{<owaECbtBL|H)2<XMP6K^YYL&L$&!o<QMrONyca^aeRi86cP5br!@d+=1#b*;;;@>n(8n7PnlNek$Qo?BU^jcC6H3"
    "Ja`JpW&SBer>0(0-45t!OV6&H2GOTe2gB63BQVOk`N1cJ+K_z^Dkoi1y(h|_5Bz3!rLR_7mCBW+ih-a8QCUz7TtLE8vrl=B~N-"
    "T_t!eXc36VYC1J$JZcBB;1OUmh|VPgsGDiF7Jm2eDzomfY!FR>!7XIXjzRjuqU-Zxm%~E%Og3__hXaX_5=dMa@B}_{v4by|w2-"
    "3uN*yZOG|3ADcPfETEhpZJwgOnl5DCrHgI&P4IDI6*MPfm<rh9Jmr(u#@S4~3pEi21L#&^|C;QZ`5FCLD4p=}GkL&mZP|?ktG=5="
    "V|oYo_?BOig7HSSt0J%L^h!brw}cpo3S&ik1z%7GtK(LDYVfQ(WEk|h7kx6c!h6#lA8V>J@fb#IKt+DdOUGp3=Lr&Bb0Qy(L^vkc"
    "gElG$Z@lRtxHUEc4@VG&i^jE;yBXYL4vPJ)9L$F^6&WzmK40~AI}fQKK!JbldFDL+$k?G~#R}ZUll8Fs)@y(Ov>Z$x@liQ`wiJf2"
    "s<m@anUIRyzdIhG&A<~(&Z=D;6hG9zK#V&ZIL(ETAw?E{x+4rvC8e!#JT9-"
    "Kpzt=~0b2Iy;ePay;ml##nEnPcCOX+Iv?nnFc!_a>9RH}-;@aIf-yT+g$|w#KFh@ApPO)+x&GSf6dXkuS2JC$;O@qmIf5UM$qY#3"
    ")s`E74a1U6}c^veYmi<1g#iK64lAw(1f<6Jd`O;B>zJ)n_tiPI#W*Ok{tRrk>b0{tXg=I^Km*T}`|34-_UZ>PieF4$IMZt6RR9W-"
    "kE$52iD^q@$pm3ZRB{pih@imVHeg^IrttNjkK%J2obneO+jbatn-"
    "1eCG5@jjRfL*G!h#zXMUQ9DgU|lWUw_A@x?^)QtQXm}8F4XtYH-"
    "(^tAi~4!$M}s3dX{7y3dIvu%|pfHY=6S7E?3SGV#0%w>ZnQHD>6-"
    "3JTt;E35B&X*3IS2ZZnZQh8x@2oWn}#?NK}!VWoFJ0>0~sZY{L*+}yO0FmqUEq_gI0Vs2pEbY?+L8-"
    "ghlPzXm&1+rTWQ$e>e#}~C6^K)?ZLnU(3ff%_bcXA6?<~fJ$I*`=w?E_ykmHoGvLF4Y{LE2DLlZA&D2%HL%_R6X}84=GA99PbZDQ"
    "bxOx)IYIMl2o(XZ{~46z}mw$KIqlCc_e9^AoLnW;KtP2i?lW6h8nmP{=}AT6WvYF<Y9lw&gsnmmF21J}Wp&azthfv3a)Q&4vIK1E"
    "r`3-(`o;sUr$Y+d}8EfS2pnHL8t?(W5804i7p@lZQw4h4VZf%Ry`XY<@wl!uAfkn&ycKSj&JL-"
    "(0<TNq!!#*ev0zS%1%x23W=gXabwgP`_ey9Z5T`2S|&7Lu&Gi2R53cjHc}-RRZ{i(~A-"
    "zXuk}TnYKnlFIn^X?7YiN?&CRKS)xI^$Fj?VO9Lcj-h$WBikYL<!cnGTrGUV`+-Sk8C?BpDOx9h^F8?qVJimMeC-"
    "<87x9qqSr7B+veC;Lk^L%GLE_^Lu6bBpyJlE_y$nKpt2VK%+wlO42@suF|>Fl!WZTg>C&5S^RkY#Uq6D^#;ERU>j#uhzJ1PI~r-"
    "3ex)3L$<$5xVXDCNA08b66V{J*wg4nVBi!hUXKRbd3Z8P1{Nt<goYsMB6pWjuBM?1Qo5T99qPaA>F22$9y<o4rpLlz|)Veik5CIj"
    "c^7*gPny%(_*?ML`kZ#oc8UWFAPvq>s+tl-$RiCWeVbZ8eOA-"
    "g>H|~YS+x4&mt90eRLb?+G_c&t&eGmMtT5}gncIjl9z}vGfSp<UUq-a4gc9F*FGL?W-"
    "SHQbeRB;kh9dDs6Q9D5Jmef>W}xO=gs5?QhAaRV}!s)4&6e`BtqQHaa#;_^eA*YVYa8G?y={h4FytL^eoo-Jh`dYq&wnRH}GLfVN"
    "#{2$L1h~LSUUEAMLLVp6>^gzRcVH!bnhoVdED;1Zg**_0EyJFhtB@?-"
    "c2yP8)jc%hgnL${>|{;9JfTh^0)XW9N=#yMr)Qqm1nCp^ltP2p0u<2`2Qt^xKHpF<D<E@awDL&<q_Oo+6*K9jNt}XOeMX&6p`UtR"
    "SQtG7Mc|4Te7#(&jQ=L!|~el#D-"
    "+b!9{h#xY0<t87>Brim9&YZ^Z>IQuhprRTX;l~cG`*k#QV3d#!z5fmtg=a+}!0~sZt6rP*w%X!cA7rM>LXO1FbJQg6MGuoNQa(5v"
    "I(`%>rXW4}|^glR5q#ZZBUwHhsa7yLw41;1!<I+$brm~ANW;mTv*TLL<@S-SCYV>?FQIST&^j*b1<1Mm|DN|N2{R(n-"
    "R7df3k;||Jt~EusBu=os8yAR3y~d0)M+aS?wbeFW5B}@&K#Ac=mZ{ZZ2F$T1=bh=hjnm(=FofeLM@Ja}fb)rbxFl0b2;iVFtgC0{"
    "X{wREbD1}jLnFY8fVese+85M7PitNbK5P9qx3p$iW6u*aYnc?z4CM29cz39Xz|6H#@ullIIPlUsqHb&!^hB)0{Kg%8TtL%}VtCYt"
    "Szk;+95qJ{101}wJQO{T&uu9<;}vo<<2eV)hJe10TDQQ_OmvzU*d!ATzyQUW**>U&3t5OHmL^D<r&^7M`(sSZ3l9WZo<t=Sk-"
    "f$aK9BD^*YjEe-T&y!9h&e9azQU@9oaLLN+{&a9f4BFvhd94$lvKu;={Y`Ih2s#Ah!i;pm3TsIF&k#!qoOTWs9~b6M>iVkL_-"
    "?TnZxund!kt2*kPyaau3`-$k|zdW1l|7?K|1u3UsRYIc-a5Rh(s8lEI=5<alhg+jd3+5kd1fflTU|BRmztjk<<U+2ryiGOAlQP-"
    "?83pt(^XYWZ;fRZV2iYoNcsji%7F%0UOD@1u2ABp1*C_8kCa-"
    "!rIVaTy39T#LvHM$7kB<TS_lb+)PW2f>jnmK^V+TJ9+Z+om<%o;`;hO-"
    "R)q3BLw1Og4<XK3QKGDG36&1Lo%G8gxc3{wz=lILpa9Chj`%c^aHKee6n+>|nShzK55SXQ0t5%N|%hzNaHs|92#6l)Nm@I~-"
    "J{Q*VKb3cFh#qdSZ2Rp$`F^`8wxRCF~X~|p{+ET}}f})v7ijIE}H1ZYP8JOnCyht(LsljN+{bP)sNAh5;hlJql5%La3RE;=D;tmv"
    ")Mu-"
    "{)d+@+mdBqHGKH#ZLuirIUm98KTQq3~Wnr&y^IzOlNmSk#3KrlKUO~(Md>8`<qWa3;)zj5xL_7FAe`VD6Eqz^_EZ%7HTxsKU@o9P"
    "@Lq>;9Cw6)1WG1<$*5!*+oTf=(zeAFK0&a*}P9*5akX0i5wt$&7{cN@l0s)f5qI&Pb2;Te4Uc?y%)C+Zd~q_&5#B)H6Ul>?rF6gG"
    "eqyFcP<hVOO2Bi*C!_|)R9AS<v{{zV>W6AV&Ilzy;>nT6|*d(NL2bY<tTOa-lGxE>>uwx^1DwwEz!G;1_G8=s#q%IL(o9WGEt67W"
    "`yShF2DDmpTXoXrVymKk7B3TMq6p@;x6ZP5ge%eD~#yy~weXP~%1!N)uu23SYoHt<g&J{%zM^cw3vJ+3b_ZyF;{pabH_t06pa+OE"
    "eiANzPE%Fa`@jt6=0`wc-snu)v*`<R^d?QvDhRLpV}+_(y~u!XqQzDDdp7l4qYAo3d?j{%Ko+3PncjKze-"
    ")cUN^tN~zdOmr*ITR9^Gr5f66v&>|uyP*LC9n9?PUbstpSW=sIW5#A}^iSR!NfXs0*(6OgSRbKPwSncf*V4%mXYMc~!-"
    "u6fvR*|vV1ye+K<1qt>I7OYXi7O_rF(=DtZB;rJj~q5qUi!~J7X@DXJlUP5_M8fQNLalbMdw<sag_WFo(uLX+?!xQ+%$CP0-"
    "`H{}R2D;VHRrooD)VexchkLme~GqB`Se4++Slo0C+;08A`TIn{2-+df-RQe*b@)9q4!A%EGe%#qAe^RPFh0-"
    "5<xdU11D9elaW8E<sl%SQT<w}^Wl#1Uw|Ul@j1{?A*WTbN!cY_j7qi<7SQ9B~Gm8@xuqeA4WkV*|4WYCed55-"
    "7rQlr9a4O%dJ+lWwMO95-"
    "sfD*QfTDdE3{W*XRo#si7D#+u+R;vv$EHA&)?r>9|sO;ft|p6iks0|rQ7X7NN2oxz@YtK3eyi%dEp!y{OBIt9{lSk6e%<CKj7MFz"
    "<WV<=t@TOeZShRC$-"
    "`;^9h=;feBoEjR7wF8wOdOf#JVzZF+dotf=zH~U^AWy;#2$0Z;AI)TlaKNG`)H?JcJphEdr$ndTpXkg$T)!`+=s?C;Cp43_qr^ML"
    "{Hd+>r~?m40=-yBg9q7l0&0NRxDe4XdO09@d1jNPUYHIfv})ceFr8-"
    "w*0@rDC9R=ncmG6Rs5oNs3zbT%iDdMc3C9qR(>My_0U&vzA2Um{Q-"
    "C81b0X;a=ZIO_bAgxxpk+TA5FevjMw)wmbiAOwJ+2ny{Hw#pPO|o#)w*B5=d@g|ely+8%$cKYj+F_l-"
    "f|K`n1)s!>%|0Wo(nT<LkXzg&t(zpmf(uHL27%ta}ql9q&ut1R}Us(+yE?xs4GA~nY*K`riYG<&ZQe2@ZoXhtlA0JREA>Wm6#EY>"
    "5VxXj+hkZ@p35)CDjw5XZer=AcKvBwl_^uh|lx$K--)$Xet~`5JdDZ1wVLhf1bpS-"
    ")5TI&^!L@h7CSgK5!n5!rEpV(1cCfWX2PD$%7Y%G=fM)%c40cXw?8_xv)5wXJC}N0ZbeM2$ngk%;r4+%E%3l2CZ7Nj4$Px>>rcRv"
    "lCgdo<(2wOiZiS^PqtRw$^g0)U;zPE-s8s&e?vJm;=5tR5B^~kd&Xjy+&H0!>WR~!UUL-"
    ";iKclG0tLXJbH>VX;c+kUL+eykE8(O<95s}4IVf%W2gJqAoRVQ1Hr3M2ZJa^B98VPJsT-"
    "d%0uxD2@T@(#Dc`Uw#lK}yqn}Fa>Q%{=G{JnaTKECV8&v@>%7;15X#CAfEuLpWLXJH2)O-"
    "{*;+1}ahSgQoET?e2;<s1>xq|9af@h)GK3oWO!?C;TTL<bApOW7KB3703C!+X2#H-!^PqD08!lbw$B4p-"
    "4Q(`*d%1?V|1Vkhk|RsjB<FT6)lLHTesC&8bqH__Y!I9~eTk}CWP!L1-"
    "ajkzoCtTbz4amT$rrNcrwil8KCi#EVyS{kCbnZB4NRUe=fTmA9A&dp6hw<EF@25y9DPi5Wlt-"
    "X#I9~{4hvSSaX}tOX**MTl+||)G;|L*^}~GGx@_r^O|9-"
    "RpJ51Cf4tAx))480$u8$iuB*_h8e7_=#Y1c6DF5Ip6Bs9d%l9Qv1NaATf6f-"
    "`Wq7^b%ANdz;;a)c<Ake>!aALe#gkO&tVvWd;R76G7`dYYGFA{*m*FyE-"
    "E>uFu**C>)%+%%hL=7OQgzqT6(G9}oinnY_%gQqc+d6m+f@15c9V={ICX^<$ZY7!N=p%P5-"
    "*unPq>DT@r@IXy>8S5{IhvySxleSS`hhy$;P2}OctpI8;6B9I78GDV8q$2=CgL}&<$OJ5ifCZ=N*{XyBpp_B|lgD@V8v&-"
    "XLPl%EsK*bUWK$ONV5#{iRGELw#(l%mdw34X@}>leI#-"
    "eI5!^<Y<+zV%rvGNdZ`m=cElR(Of(98bh@tveu3mu<!jycGhjiemKKbZV{$a@+Zo4?c4kg4ft0lyPMiy@X8{~o*ikR-"
    "Q$T8d30KS?j}x}l&i_<^xos0eIrf99p2j2DlohG-2@}}x}ungs}bzli}6m4h><Z3vVr*+0)znQsY71v_9Fa-En#&vD%<M&ciS63d"
    "_k{Aq<#MmL?_96V~2H4Pf}$>iM*8APj17AH;iW!V!p43*LjRyn~%K&;{XK~SIP}VPEXq<P+X+QW>-"
    "g{9G`><anq74%=SG5<;^p2d@mCQka{$MY6GC!qUR@uJI7h$5)C164LQH*k_G?;1m$--"
    "@3zegoge^ln9Ky<_}9kB%xvk~O?oSZOA$nyw#hE=``>4Dontn2trM5g2H)FAIH_4g0bN$RJ%ms4#jP<D+%yZ_Ae>!xpIi(cINYSf"
    "ic}Diqyrm;2Ltoz04O1Ks=p`1&n_rEyL9(X|M&Xj8X=}7A*m5xrX9hxC7!ob&wG+uqpx`;!I4IZ?&r+Lux=JEE??rWx?{C||JGuU"
    "g`QdF(QK0=O9D>yXp*~#b|^wL@HP(!#Abt@B9+c40>Xu+sIX<e%6{TQQ1QA7_r0C;?n)iOlh$kN$$rKd1Vm(8)uJ7yPx8s6{$3r^"
    "lsFI)zqnA&)?T`Hw{U&#iKvPeDr#eCScNBz;)~MwTxZ>VM%WdThB;dvK0z0O(`yoI>&4&nyXpuKqnaw(p3%<|46;|iHMd0+7DD3o"
    "<qu}1_a?l9hp5+!q3T?!(%Q!w2D5=~Ww#%`Sps!Uod{vuI*es;T}eyX#`2VCH*ZC-"
    "K((FH6}WqT7s=Ycue1ulvsEW?&O5wkt3y%(TsUX$7GGx&)YUips}27Y_z>P}BN??ifevonlIbu_$}>=`N0zA}GEd_=7L}^?@oUuu"
    "yfbUXHwSKS#zb<}33=ut>>F{OSJh=3PExW_n$KnwCF%4kLnKc+k%6{V!(S@%*SZk`$1J}LJ|qETXOg9t_6hQLLE8qQJViuiK0$R+"
    "mVO?`m|WEAcfxvm(Sucx&UZbUcxZtx+CqfmHo`OVMn1+n%Yu6*<hC+L#!KSsCP&qGav3c0r`E@AqiD(X##=Q`Y9DFx+H834I^LI*"
    "mnc2athU#{AGSn?|6UFHR(8BRnwc`3ZK&n2-y**~OR~JBR3w@67&Zpr3N@c;^a^gYgYEO|`@o$vcPA(C9-zB<^D+@}@_U9#Ce-"
    "3MSL7a?snCiNyNE&B39FLT_)jC(l5=#>$=>UyfM5?-8y@d;A<-"
    "+DnQeK_Yf~VZ1r>`XpvQcKW75#fx8AWo+tQ;Hghu@PK>9vg!a`F(8T37UN~J=h?7G9ST<bSit<XpTu-"
    "r2~ur^+DS!jeOTvR>H8OtlbPN)w}d_qiA+`XPN`;As%&yaqoI)PN13~uY>x7qiA<%9=tPWMY#vnOX4&wSACcNPIrd-wF<cOQs4b#"
    "))@m@{z#hJIsV8@Bpc@=i7`Q`?#!`cf11WFlHdIzHaDwiVkKLA;xZae15>EzeBPQrXF)LYN((rvN6*?tY!w<Cz>0VxJTz5p3l!I8"
    "YhU3;R-g86rGzk%*N{J4@g-R>6tzwDut6Wo~jzwGQ+UAY8Nm&f8491i7@xqiw)Sr(KV|y|jg<Olu110hN#DJLS*-"
    ">RvM!3G8S57$F+GvD4Rej#{llR6s4qmwf=NxMjRxj5Ob(A%8AL)A9Zt)l@hV0ijj3;t<t>s}$yfk-"
    "r2?>_ndgz%9~LEvc@#_t4>aaHC!UyAIkrL)R;bii3@?{o2Cdd^5eO$n}G6#?`yD!BQ6y1hJ|0{*48W=yOzPNCh36#ZQTAu;_35kt"
    "Rb3;{$N)SCQ@W_0SakPBO~0x*h4rjxDZF`^$b!{HGUQjos@6Y?+E|m4@5n2Bm)iIJc(Dm4OT9r5PC8DAU*~Bq#j-"
    "@ppAx247oTcuQj&)<CqC=TW5|AIpK<HA&C2^wM3MAk-0G%lk~%?yTy-C?-@F$-"
    "$(<(b;3nH@XAT$wc_JT+y5X=a1BBIr|02LPeJ2J6$_KNZ9k^#XFuswUS|ZmA1QOfav~F>0i;e%+<#9hWv8|Fx12OU~?*yc#p)ZBR"
    "Om<@8D3sqMk#W9gyI1%;+RduM_(;jVV!9*j}$;Ktk#7u}fJ^gEZuD`CK(brv{7M-"
    "b??i!az}rTNXNfV6b$?qFHB*<i;ixqkCT`F?uuokHI^`<(a));Q`dsK0LuA40oGPrE^awNDKf!J#WiUHN;=wU^|`KTXm%RJfao9-"
    "pPSCPRVS6e&+6di7UpV9bbm?v9AyAZYFyGjWS=@u*Ko#^ZWgsZQ3^AV&qUrw(`|sUk2W1j}s^agWq59rpq~k8*2Eb7Z^)v`y)`H4"
    "bj7DI=*wuKJ)iwy3=gJd%VV0zvjGNWRuM3LH(0dU5|Kw9~S&p{jX!~X~ZET`neR;bH4WgtfTS~ztCOlu>=$L<c#mV^#RXfgL}lY("
    "dsr;5iF`wy{mu5X$^R|VuJ}vbE{e78t5~DLWqk1qIPf6KNn1Z9g-"
    "8usguf^77s&reTTUFd$?#r$+*_1AWp{e!s>Z$$jrMpwQZnk0S#}C9wD%{)4=i_BzvfE!ED#Ys9fMQ1IC_o^c^U^##?<P)B1UO=ZK"
    "CMI~?-"
    "R{Tc7gs!=80<W3BZh;7+nfs>Wbe2>%ucT6SR8SqRpQO40db!d#9^#ueZ0jW!XJOvMZ&&DH16bE2#_ZZcS{=akGxY$*CZ*_-"
    "2P5`au*Cl+$z<Xoz$*{Jyt0ZwX0SSSmm!yt)J6kKo_DTgWsV!r=>r9%5uILGo5)ZC~cG?p8J`dPMEd<dk;EssdJ4~$d%tK`Nr8EE"
    "+OqzSyuQM-"
    ")xI#kgCw#v0*+6^jp^zxiYucvUG>baJ6+b&zmmPKI4+8n?+&UZicoN=cXo=!&z)v*Fx}kI2<+i*RlC}n?I`nf5Q0UAO;_cqgq*j+"
    "n(S3gg57c2wO={GsD7-%|<*LqR^sFR)qxlLyrrA2S+zI34Qf&mlxf^T{b-"
    "K9vn>Zs@557Bhcn$z2St=hB#U_f8N+n5%O6dF+7$6L(0BF|lY0BPch<|^P?{dcTWY|ys9DnC5OEf3f9!Y~x^-"
    "L{JyjUZ{!>0V(q^g_9n`Ib&_JkvsYsPmIje}?|+unEc43UA@ny8xKBNDcj0Ed;}0{VSs!!Srj#bbIxUpnPRLcr<cLZr{5y#-"
    "7kQ?ZY$hD|_PzfWV2vaEB;pMbAOu1S$v`RX3=-"
    "77qfw|AK^3w3HcDenf1t0VU{)An~*T)z7!r2c~rp=SinGv2%P3_h|+FcHmb)Z*Fbvpcg>LWg6rJKo3+*j(^d-"
    "@%fYrj+*GF0VVeJ&yk+{?m8H8;oXDznG_uFHGKos)onm`hKDWRqWw6KfT<I<42a=6~1*x+?c961ck5<h$~qfo`}0dxTW!mbwSN5A"
    "bGZN#~p^jZm8Q{6(cjZomu2D?WSDpBlc<_o+;R&3DRbSU9q<y6Uy3jKJbCI8!U>DGvx7o!XuVYXacQTwwz(TcGT}a2|BlqowT&0G"
    "(c<_*Zcu?XC~wT4(9|<tpVgZK?@4qaRWXfkwOM#{eBSPEoLSL1Y3g&!cE+g(r(aR;{qbwEr=JdQq5=f1cY2m2fB{JR&62)`wa=A?"
    "5lYXr1WK|GPFWQonYuX9c>NXg}mZfJM5Mx&zgQWb}^{T_-"
    "l<xe~Co*>dj`LM<fsEEwyW)dSv}tK{$JVx;l$*>wP+hv(`XY1qjgiJWGiQ0>T3^e#2hbTJRc<z!9iz8}(IY((#D)Y0&FV46ES{f#"
    "e#}52VPJgitMFT}?itCAJ>(t7>(%iK{7=SdIZ<J$WV!Jv<<7A0TP*o;@o2o<?WXqiGO8Qfo7nPt@}3mu4AB?n$(QwAo+Xo_|H#Zs"
    "yuLL*3s7jbPpNR|8suC9Q4<X{H0TX>`MWleKewMZ>NUl@of*xel0n6)^Ta=C1KP$7ZOVUz<G<QKk{}+xjaK+}|0-"
    "GZ%5v)6XJ(1o?xNz~mRuKv|wfCe(XUWGzi3)a+zKynI$i_;=6QYk{a1DEppFPxpTGo@7Mw>O5^dRCDYB;;ZnM&Kr$j#wx?Sy~Tdy"
    "4Dfl$)2y&-mLJu)aJIdfxmkRI0TI#cU*;<eU&iPFO#L}--"
    "U^g0RYo;&C|yT6OA;ed{rORi{8nJA3J@^CAT1HM+m=T|;DjK)cc?U5vptWvzT_~qN$jp$Bq5c@(qR*>0SU-"
    "&8j)wF^KWco_6K((YpK<i(3E``O&>P&IN2R7H!F$Gf-"
    "U!7Q4k86LajyMezKsnL1Kwn_IF{d5*rJThOr$f;BDjISqPVyfJ77o7uP!6C9E#PW>)pO*Pe*2H&RnJ@6pX7rw7Y#Ak)5ZivtAzBj"
    "L6U3*A{x-k?M$dpF3cSH}zcH_sVqfh#Czfp@A5&xKn_o4fb+)Vg&EW08AjZ6C-"
    "*kSq7dQGUOpj9`1DzpzM{H4Dj9Or6P1=>cgnBtc+~1Q~Ym#(J9&XLrotN6aS+7^Am(`xN?ppqYi;TdSuH7lLiJjx)3=6PVN0F5KV"
    "KCMKGmJGUUl;M$G`7}qN;({c^0-?5^pNc_uO{?yNO*Rj);Y>@Dd>{7A5eVz@9AFt2vV!FFE-gqRPwYD$gEWTm(BUjyr?aZRXB<~o"
    "x`j_@RC=m>$8$A+v+LbYxa^P<H67ujJrr26o0wmX9?U#U<hfDOL@+Ss1aKaYXfcs@!rIyRny3x$+aTHoPe9R!Nh-|0dKfPv-"
    ">Vk}JRm1jut+p@8-"
    "6a4x2DNczAWv)EA$BjmjLhvN6tWimDP51aX0O(gty#S(AndN!1`cGxjD4icpYe)zOT}>Q*UY%xrzD;a)!$Y6XM#$yh?E-"
    "7a7?1}d?I1gv^4Krqo#^hAvba=Y`*NUf{&|lP2dzuMJo@8#NY{!eye=sP7vJ^A?1oby7GkeSdwJuGTX0CX1H`;o&qkty|d&(aLvS"
    "qTCPW)Oinr_Q}(BW&*gq;d)Fhc{y@IsBP6`Kw-"
    "J;ucnPd#JK5`6q|Y!DRP!^nwhkg6@8#z|O3M<Yww{CP^!M^?{eCHUQa0~bLzXy6+yTiI8uT>S#8I79W?Ege+F{u;{m#$P6Yi-"
    "v2=a}aXQnJK7#lIVwr$EFqtL`=?*+;0B#?yZ8g@+~yoR6r`*<BXPrL7p!_DUOBG=Yw^$g8cVP`^@yTtkZ6+vcSsIQav9S2UxvNxS"
    ")bX@8U1&{ydxWDkATm0GaZ!YW>L%eiQ=U?wVgOM^w;numE%a(oZud2=Zvp=`kxwdHQIQLt#?rb=cA8$Z6CNXC4)sX#?FEVBy>(=<"
    "bqcH7W)HASWW~(>q7(MO&9z&yf{ER#U45A<SeXe!IA(==slwo!Zky7dED~jFGxV;<fM7wfk-o#d0r%Q#-"
    "hk_S_bCW@8rRg0gJPee9O`ncDG{bI5_Jp|do1SwQo#s*uH1s`_uX`jgBEseO8c!pBU(~DmvSXU3>|%|3#Kd&WrYl(d*G_R;&E$#D"
    ">(YVXZ5OsHb4lXo!HGYIsIBG_;KV&~Hl)ANKA&0H_dnCZ-"
    "4PluO$?3W=Y2BwezMK(Z<vx&Z<^^#cLEPLZ|rLi*>;_dsIS`V<o4BkQUCu$YMq(IWWV!_iXJ&l!0t|Nitrj7uVU}&RY`KxnmeOt$"
    "%kkM8%7%A?H*I(?=d<mNV-G}K?|#s7tV<uY(}{v5<@ptDsGa|H!67h$h%EnZ|(#&N-Ce^w5f;vG(nx@HSrpz;|K(&-"
    "uXRMDk&=2Pk9D5s$LMmu(k6v7PyX0X@B3+_mwKt6aY<NLMo95F^DXE?z58RWD1z7ky$iM2_l7TA0qUsU=g%Z^|x8oT1W=}4Lu3_z"
    "}4a1S?{Ux-r#1cT-lR-eMDV;b{6H49bU4;_q+1ZelHn}&XIxBXPBaGqHi=nUW=?k!R0iQB!rS%Wj<^qV|C_kEna;zUUc1eFt@Wo$"
    "%AgU4A#+I{;l}R{>ZlV_~vET8G=TketoWn^IPPz?+k6EMNogk&O9wY^j`tPVY2OgLz`BaKGQ~Q8r<^(02BS{Gp?oH2~z$$`RK5!B"
    "j4VS1UI_9&JssZI`IjaU_H$mWh2ej>$WGE1wXiT2rZTut*dg0J_#`;mL5>)5jT<Pfqruod-"
    "C2$S4_8$X4CQqL}001>n|)a9C$b;>7%WgzgztGxBEk~P_F5`UzfWs{Gx5}TpW=AF*c9`whNq!ZmEY-"
    ">8Qj3{N4e5XC6nyNJCWTV8_}c#F2w3RXQK{pkq2B)4e9L$u!%bA4YjC+VT?d@Dl2|gXJDt(t20)&3Y6jh{xZvmna)6)mbW+r+!(>"
    "tJ&i@RBo>ld1rsJOX#6XksoQ{dg?~&AP+vw#{Q8ae$dr{w+~$lC?9j-4pg<r%5E&^0)|5e&sD*-"
    "ufce1c}e&v00$!;LH5Z2*avMwHry^|%U#U@@7WGZq_7pQF~w^jas)AWwI>(`govGodvAU$$HwX3O@eX(O<#iZz@a6(m5oLntt$n?"
    "?b?!9K&@zGz8#ZG`9zunFMJ;}<5j-~d%P#8qvi7+JbL3T<f6XER>Ws|$TzuQ)n1F@iQxC~nq=w$36g{hQ@o|2Qg|af5Q6t-"
    "m~?C6XGPj?Lt^#3WID5WOna7GQaFb#aa94pFzM`HexKs8-mRlD8BP%vMoL4a43pBp<Z3MzI`VTm-"
    "3k>+BN0;49UNq<^YQ|(QHHgG#rFA%2Mm+El;)Q{3@qE-zif>RB9J4elWsN%01n)6VD95nzbFQ|*Z7aGQ3CBrJW*=x`5;=602LB$u"
    "7I{dHhQZ7GZ<XE{=3Xkem}XF^l63l9(s|74=JkS08@EMWOJ!k+s?^hk^4bYN}zTof=R6KGNh1#;~tdR_}%-"
    "Y(%e?Bp3)>HH>G&5Xrv6@>q;{BL9zF;zO|DC`y!)s8hIRh#Jv`3|4yV~=K}mLq&L|q?(vS}VmUt(+rF(KWYiGmOboXIg9bI1f2g9"
    "X1U@y`s2Lz#Q=6{c3xb%5J7TfG_NYwgW6A8weD-"
    "L7Jnt=s(i3Mv100Y=%^)IHA9pM`QIMJ8H`T~YRcl1L2hdri!iKitHxazoL>cpR<Qh|g{#BMJr4MVG^XLHqMd*=};!e<cpGAKCXNk"
    "#`IemOTRpskywqwSJ=3Uh1Rg$&b?=S)rK+KC!S%7qvSA}M+XHg=ud!<uXu#N^PP*s|1KJXeSmYdu9GzZXxWSBz<>k>J8tsmt2h^3"
    "9UQzzUC=R3Jk7@f;-&|bf@w%Y-"
    "|^<`MO?EGdX&|Q9HKaqj4k#Bcwq`d5C4$58+y(F1J57Lz^jqzUd!e+vA)S2m6Wy%T>u<=#^o(#RFj~#L5u5{-"
    "<Z`6rxxTb+ezg1{H@1U*n|M3{-<ZXYUrCs_hl*hgAGg!ZienVyQK5ei|oe)PMK;R@kh)f{IuXTB<<Fg=NNd!4RlEbUp@*ywD-"
    "e~E@0g;&p;QQ3WoxLIwO!oShPGVC0*6?B?GaX*PM+V6G(R|M>@3_VSTuDv)pJr@iBkFN_>vY<AgN-aO(C&3iunxJ+R+H#+8#by)$"
    "k${$`2V^s<C7H$qp4q7;#09%Sn<At-u=YiU`OnWdHN5HnZTIqvNLJm7$B?f-"
    "!Oe0e}f_e0z;3eRvuO?Ky{pdiO_1vEv?xu@N}2}JYj1}?g%?dr|u)wi2>!B)6FfKieEC$F}Zb8+M_dmT+aP(ZmBVCG<GND(?{mP0"
    "CQz~TNz|(^*EdvG>@ZDKTvRQrgrY?#+fFU$wWUxn$(YBUGWXRv2e6H2XsG=l2WJZQ9^jpo!2=@N#+iT==?pIBr&a;$MZ`_<>pDdJ"
    "JyueN@CnI?xi4_mWq`&r#(9uK%j!<PSR;cS9Invexd7!Yb_fC_}oyl?=96y3=)8o9X(S}5SDE};T$}y(7UUYpzUd}3nA{Bc7`9ey"
    "@hvZ=0q?7Q<g>(3Xm+@FyuPA^IW5yTr1nM*Ue!%858-"
    "s>uaEiJtBS87LG)HOGKuHHU!F=Tkm%mR8|37R*PaSC|628Kl_9dNz_a5@9XGnJjxRwkI@W^Z%;*KCAiy84|%n0X7^#?Dk1}rEx&U"
    "T0MMi8U3%6c(M#OUli_`rIT^M3Y7?{FwPNTqzoQR>Y%dNB@e=}is@0Y{3Or=SiM0MKQ@5g;E?dW8yjY^I48O0C&syVSjBsH!MXuW"
    "p1c$Rn*B`r`Bi~-D&r6!3jK}8OR3=E0XW!8RoY61%&Ma6Oy%N$n161;NI~*n2U<nxN{s};^*NbGKBA^>g{Oqwby-"
    "{9_q1>*+@5jr(G(przCPGr6m0FUkS)9@gKj#2Z*VH)`8oqV_I%aJFQvE*1?3Q0Xv^^?r7?#>W($Eu;d%CX4KiK2EESPwW@qQ14JX"
    "ZW0Fy!jdG?91V)ZWo$D=XkW(<Hb(=cc2oa_gVzgRYj)28nH?91d|F*WVA9BsV63vyGXuUo6$}VdPuPod!p$#AUJ@w!68C;Y7aQ&z"
    "`{Hu_QBjFrt}E-cNR%d$X>h%?aw%0DNEi_!h0ZM(Xj7zCV+NJIlA{Zn{nKHQ#_gOp1EMhT5Vhcv7hKKLb!eiQx}zk-Vj<hmg4W`7"
    "St<BtyugVt0VMX+y|pw|=XqNA|pg9y*Wv`m#-"
    "*=qaw`%S|}?YrSD|t5_S=C5&46Pq()r+>!SvU3E{Yie_Y_jF_P7zM@ok1}1ogzizxuo3?@G2Z?f92a0gV%jJrjJJ)Go%(Ny0c^)M"
    "@9yzMVu1o1Rh_L+*BR%8Bx+|>bK%3b_dlqgAXF}s^+M4{o2LFhirCjY{iu#alz&>Vg(#sL>)S>&Zm#tBg=EI;+ZBQ#v#~&>)oyXm"
    "C9X$(fxk>hO6){H*2_llH@_RBBR3dzWcSL5>ussPF%7B@kx|eFqd#;HmS53W=d8b&(<6D@2a|8O(6#G-"
    "ElD<+llaH(kasABhq^nMs#!bZEoVTBgtnIW&7r$4*K=~cqM5ZEYc*BD-2McpBaj|rfqx6>XnmU7Z`{v~Ce{YRB&pZS$x1aqZTX-"
    "&AhkHvUrM4eAerMXPI9!ugkg(#lS8CK&l)UbHJpXzsBxo_LAKtZdG%d&qNYOf?5io}9DObUor7ex2&?IE<>aH@pV|U4uG>Hj$I8B"
    "wgH?o&2^`b`NN+J?~a6XG2mJ}f1AtG*Y_0DJbgy1u!w&Vau_v`DMEw6pwSv}ahPXMo}Ok&B5L*5VCK?Pz1m2kLL%Y|7GZ0;UZX1`"
    "%?GVM%^-$ZR!z69e2+oFB=I*VTG{*Y>~<C^`GuIEX@RKBv`jRx4{0Tf2vpZFc(`;@7*V@O~da-GdeAq$SX;<IVwx<p=+;A}K|{6*"
    "H|J+Oguhq|N+mOwD219<vQsfExJLV@P>Xf~wI*6L03S&sY~HGsw%=4@YFGcoj%^KXfkq@R57s(4!xhTii3q*@u>u93I4);@Ygs3t"
    "m$wFh(waXI3beQL(GrOETk)ULQ=XsHUKvD(96uU$ph`&(N1v4U$|ucXL11h&b*rDzfZMep_5^sL*=u-?op3No;rk({YepNVL-"
    "B_e=d<m;{@v|6Gk$D7<;JL{}V?OLO^+0Vux-jT=WTZ?$F4PRs`_De!;+ib0cJtJRr_PoQDIAd*A2PY+PTq7&fRgIhX=1?ly6EkdW"
    "JAm4N@~KJWIJ9h@qSx)cF@6vvwi0Sdm%`qAx$Z{CnIO=Y4Cq{^+*ra&ar<uLJ*9JXaH3l5qd8Z(?eV*)E1p(38sI1D!vDQT4-"
    "}9<Xy|-fk4N~Eb{GQQ$d>yZ1pK#xCD63mDn;>}&UYOpv(a_}^G*};rp_pa5TL|HKb2SaQnkx2?v-"
    ">Iw%3pfyh|wR{^OISf%Nd&0<4;>01%n{2gfH_s{Q4YRG#iLGygVK`F*=j4W0=>82L!uy;Pmfaw)9f&3>KzdAGqDK96$#`BD%qNi*"
    "{<+3P-esCqTpK9}J-+QT<`FyriYa;zqd$(^#v>;^@#7{kbRcbZDo346-"
    "T;Oja?R(y#1c`Gu0bH2gktVYkmt8@hIl6b+z1&|%Dr^9O!G?&}BM>XU?YNyn9drwz^M-"
    "yscDB3gW1P8`nIGLiJF8|XIdu;=Xf{pC=u3mr4QAcPI_UpRG>m5d1oarxlrmC@TALcF5hFemiYwhEar_KA<C(OQmZLz*FD|bn*$X"
    "%cyFh8#@H#N<VoaddwV^U}Ag3UQp4Y}icetv}J@+RGA90Gf#isezb#};all4vk$qqF_am-"
    "u7jK9$iJes9Z|4f*7ePthh~_k*R4OJ?ch;;W*#)KyPYyckv=x)TVm4wg=y9)m*MX`KhSmgE}PIHHnT-Oy27nQJaRG!9cTK!rs19J"
    "5PirpYXG{h&FxySc?kcE`kNwvgq#$7gGdd@@^<qo*gT@EZY^8w7x?p|baiAq5EVO`?a!<`)H$@^C4dfrmu_T8p^*+Uu&@x48e;*s"
    "0!zD6jc=VTQp*mU+G)a&||}Sug}9SEoTNn@6pAY+%CidEPow8VEw+<WlwH*1qgaY}?-4zQ2%6X~}$^-"
    "=t>EVPpVAIy#M^%H&D*r1$XcT=$ryv*9$8Y7f{|Ls9nk`GE9$NAhZ9+&<fvkMo*Q>dnG~>(v_RsijW7xsiu4FrB3`eb*0MN&Zd0p"
    "o1S&5~ApvbQRj5+4se<MQ3jN-"
    "C<rDoGTfev|jxRpiW)Y&o6_g67}F4W1>g+n5nQTWr<MpT{;NuPLPt0<ee;*v(S@a91L23WzTMhTvIbV!mTGs8st~ooIm^#)^TN%D"
    "yxFXXF+n;nH-"
    "$H5cy(x4c6$fL;a^#y%Z@HmNU%d{&*7?lG8g1FSP$7G;3?JrtzD?oe3?Y)^g2xk@>9z7@v7_h45Zc;=iL#G@{YI9U(?^g4CL!r*`"
    "j5XOm0<m#FKSj<EC$`WEtyGB7MT|J6M3<F`Dk_o>~g#?yv8__w7j1zrs(a9-"
    "yluQA&$b_}PR*<4E|$o*7tR~V|pipBi2bB(gASBm^6$Qaz0_da6c2Qpn@Es*+X7>)t?6P?3TXWE{zLejaPbzD$E#FC>ua6IqQt<@"
    "IK6E&6%^t~2%<xUDTZOV%SQE$95RY{Tk;xlN3K>7T?@8=F($v0IgNZW1ZyaV;f;q$IFiqg<|#8nEyML+THl3Y^Y+^sEaz@WyUI#@"
    "XyWAs(^__m;FD|XxIE{=DvHCvsiDnG;=Q0(`ZhHa1hKWz6_0c;t&`+OxF_g?W9+qa#$W)KbP2U?OBVS@1N<I~*{Dw6#H6Fr?BN-"
    "?{l&Sz<iu)Os`w<fB0mbvD37Dg7GbK`OgJ))sd&`Nt2@9E9A><aD`IL4RA8xX*4_WQ2ZC!$DSaBuzASYR3H<#Z3FS%J-"
    "o1Zdb;6MLp5FQz45HAu`R$y3;a18Uf*96O<sOoqm{mHKHq6f5!RDXShkl~A%4g^c%Sl-BI`xOji|v2~H3Au-"
    "_;EiDF1G$?1Y;3T<4GpE!F;Pn{qRz`A?={XD($0Bq3=}N=_$5fEr^~t4Y^pa<?J`($gEz!@8PUKnrI>#Ds{GMTKy_Jk-"
    "Pa;)>EcP8^CSnpQ-hCNruMJLAT;hk8IRHQf_dO8$_4vYnlmu6P^Y!E3dMj^fm1|+~COgWAdP~mZ0O$#;wwNaGqAphh5eKLF*c**)"
    "&1O7z9wwLdvbC!jI*NV4m)Ns@2}NgUEk}#mmaGH<yz@|PQ<uEif5W2L)3ggS7A?mQb!YcnHa%~E{OIoebtp8*Yu=;4ZA+r13SgU<"
    "$*ONjP|bG2VwvdBI~IBi8bY>v-e7j5IH-Fc=HeDSV0p1dtDIl@_Y@maX>HmVZ4gRFiNa+{S{ah^;t<SUcSS!imqVR^%GT(b_X6SB"
    "aN`Zjs%wQS8!p=Tt89!V`ZM;7+}6;t<{m@>Vowc7x_4&+X8_Ss?h`vIj%H8H*LEH4ec@rS<9svw?+@ro@X)O`H|S_@Q14zdaO`{*"
    ">Vbr6B|~I_munrj2j@C%?iKg^9tUR=ql_b0r{TkS*SkS=Pr=4aTU$>-"
    "t+o{w#iqIukwL!u85CI^82N_oq1!zMeHKLdp4{<~O>xGPnmzFlu5D^UOhZt(GU}?dfi>Cj3}g@Z`AB)JyId!#(6>{KsJlgC9E0T%"
    "7%B1AC~)bgC*6lY>cq2^@)#??;cGAp@ii%}Ourl=uP?1c39&2uJ}=6jYp^_b7}T0q_hi3vdxFA$59*dpZ57YHs)!enTZ(HIhKYF!"
    "lwfCw^I@{+TX;8T!CxV6S9c>C<oOsa^8GRlJ*M&j&A;+m-"
    "eZnIao931y)7*|rL5AB6dW3Wo&a9S8Bo+f*^T3SH*HpWIHh{W!w^K@;mKHzrqY`gXQv`IqF18!rq@~FH)-"
    "km>|4_|n<2k*q67YBTDD|k-"
    "9_%rEe_s=kx{=CU4}XZ4uRsmxd0oD@Rmhzn7t)hpoEWu`1eP)FHU|SfPk~1sl7*IEFz8?de@nHz~@IfaW8G=-"
    "FXfZ8)v24qsl9iEYF*8<L=*%EGez6HJr-"
    "?0x|FI#JPdYrncqAO2`GON~<_rdC@1O(W9g6d7*6!S0C`^QLUYxKlc4gtvmBN5%qgKEza)}DI8`F#Ai1taP=Hdv=&Cr%Y-"
    "@eVOvdQD7X&0PPZhlcVlMk<|4Q68Y-jVLoNKAes3XyhvG+6?O)MwbR;-"
    "}nNS=)@V9R<FU}QA9f5w0WVGIfZON3V(~o2@!3}Tz<snF!1aIz;Wnu)eP7?t^O8rz~rcYtX`H?`hMg_v27`z;bsjW0^&y3aAzV`;"
    "D3JPv|L^v^=HCEfQ@AOJDvJ5Ih4U&=5X#_F@PbFBqwj8(3z+NJ~SmKQ-"
    "D6ze&q;PuB5?Y0V{JWEoToVh?j9V+<xH>I|tM1Q&GXa6XNmUbjuZ|}&l};nAiz!dDABO#_POE2)6PlEqd8oPVyJ_?mx*}=u)f?<P"
    "GhX<F_xOMm@Lw#Y@3VbUCsPjT?aW|}zBS#BLnLU<K3tB>q>&^Dh4yb1#ma=`=p<+bZ(bL}!?nSKx&ueH-qc}OCx7&b*3ag8M_Wdo"
    "jCPormyh|uP=}lL`@a~o*Yy%;1IfD1f18OKL){WT{Mk)Ia#?f>=BEAQ2)Wrwurs81=ctVu&?~rGH??NcQ*;o$M&8vk0<~Otd5oI-"
    "RXSXX<b)4UoR+wTJ>k~}U$J>s2X5eV=XNp5BrER9DkbCdA-;=jZH}n2!<C(}W3og1zXF*iU|Hjy0kXYoRh|);Gew-#o>i-"
    "*NI12CtUkv;`C=@PN^@5Vg{%5^1F&5uB60i1N>|Cu?<!^2SSS=%p#{}@thgi8Nh0$7cG{L&JA#2F*LWKr-"
    "V8PGBca}Ft&dN`hc#DFb2B~hY95{v$s2F&!Gg0ma~Jw8XUs(DaV9}^8)83Ul_3lT$vf=7BlfY3d!Obou)&o@@L}UAttdT+zdq!uY"
    "dqlAe7`&7*ponApM`iQXA@gv{o0gAagf+~O??YAnMARq<c)mcjoodzllj(S4q$0vYmN{A<<12NEe`Y9%6Iz^Lmt)p9RE;F8c@2fg"
    "6eLC*yK--O4VX_?Sevu@Psm$HBy~y$$mZbo1#97=tcuVNdkfTm@yZRf#BV=&pF#?kh!Z(MbaiZ`;xvH0f#}bnGb=?8FG42p)<N5)"
    "!kBSS$wWd8`!e}e#5-F`dH&Ge6M6J5O}^-h$rlO52x(+-"
    "d84<uAgC6eDl3`q`E!v9ebTz>{6^4)R?ZVsSF29AOqkr0NZ9G5=YUVg<{d!vQBethvs`Gp($_t%LIn3X>>DS>Zq3D*IirkUm<?x0"
    "&1Rm6WzPDdxpj8sf7elJRN!5-"
    "3jGmVpv%MUEd}Jin+KBrfVI4Cl0IOy_#!l6pf_6I9UKB|0!Hy;(7q($<jN#j`H4a(g?bD7Ty#`>0VsWPUG~ty6((KXX9c!%%hlcX"
    "`KKduNxtSxXHh0=QSXP7Wa-"
    "R9BBpwVoNIRm~up+!63Q{eGuIaExJm7=u(`8Gm!}SpHW42e6>L$${r~A)`CMLCIgBJ$0|dOdyVp2w9!m^N#Y-"
    ">UT*pp%Kxq@zf6RBYlnfyKk6k)+WE-w&8%P=Vh3#K9rCGAl(6gjyiiclAx1S-"
    "QBL?TiXF=Z!lS<T&t$w(H)@qzLB#}og^hi+7g8N#xvfE?CGS##PKdJUaMo18bLmL9os98X_LGIhC9$Q0s+*5`XS08JB__$4Q%=Q^"
    "TXwE<FZvodiwRjDUpz=Q4B$Ouhz<YEFiU0BRU)1tevRa@?F(X1U=z^?IX;!K5j)c_ZPVWS*Vl3mO(_nSzU)lXX)QS|wsH+1L{OC3"
    "ZeB$|CSghbdW6E;+stblc@qG&T4RVbhF98$I40`5NtmwbH(=&9-VFy(@@@ktdi#&O@}TV5v-"
    "Rc|<&PCccwi5gt%nVmcqI%RySudL^rYzKmR?VL6>l$bJ*`zAa~kB5CfzD%T1jGp-"
    "V$6_Vky!tvkt?6yM2`w#9N^b?vc=8*ZKO`q~E?hv(1mH#E5nq!{cexb(Amqc_y?JyK;>~jk2!<t3Fj+ZM2lVLF0F{-"
    "|Wr1fl@!`WRpvBU?5iBk>qYQc~NfmvhFJ7R%YGrGOKkI99h36#(i25*_qi``M+&@vXOIA=>3gK-"
    ">%KWIoyJ4nUE=uV#Uw)XI@!7M0E-HUWV^=32kBdR%h;gK{!$|&58ebXnP9c>@rHDRcI#Bl}OppgJ@MscWefBMn#LN+9R}A_J-"
    "I7%%AbhI245O+TXY4d5MGPk~3;jIN@HTvw9)G%6;lly|^R?zuCa&*rrHDC=RX;IpRl(&6O;<O+boO#uZeB5oZdCKhr;jyxS#QcDY"
    "DQIvsHkJ=s072va%n4Ys01CR%0eTKK2Ug7p`&vd6_d!inF}<o(fn?d=J{?l3Y5xaRtp;Ylz=H?sx1I`nx<hL0j$_QOB@WFOVT(3M"
    "o#!2@8G9H-$pwlg&Uy=gp}wIjeuiS`^Mu(=Iv-"
    "P*2L$nTaTAax|h$x85=cZzy1;_+fz)_T={t#;lMZCjQ2npqo|9OXl=in!iZ1V3=&-"
    "XbgRvqLI&$~8F<6j57p8)^5aUQ2C9j;F%hrXPFG?3y-am}d3knWI|BIy`y`5d*40V@rd~Cz*O_uvmC%m?f3!rtOS%zUONm<g__n-"
    "u||Czgg&T%=cRh8FB^6SM9gC)+G+_EvbevqL=c`{g+*&qVp}bl#%Tf0_ENzfd^0z9eg4N+L&N367Uml?bVg1AxJDJl0W9niR_cpG"
    "y<@18)l%b+<QB&WHV=zX?ulBw|C&U6zDxkH0qK1D4sI+5pvawoR~^}s4E=R>U1@?VhBY%hlo`Fu^;C2Pt!xTnP&@EdW#^YNI`GzN"
    "p*q}ewKx9A7!qe?-~7S4C^7Q21(vNB9jizL93W{tRn<aqhk6ko!)~2`8!_8ChwM-s%%7u;#;gU7ls_A+Z7;jkD-"
    "$K3q_31wY*Auh%Ct}u#+iFM1+~`#5@LkX%l{1JQ$#qB_Kbsn{ngq=%!TTdNSJk%ipA@q+U-"
    "XPnj%!hh&c;4c|g2Wj(5j^2qKShCRWMb=MZa<FY0(*KS746|Ga!8IJ-Ml)}K~&$&1NSb!ulWzzn8n{A47$9$aNVA8{`L@z|u8slI"
    "WVRBaK%Z*w}*DP;l4OCBHW1SWUfcBm-"
    "w`oMrlGe<;e{YF7#L<swkPps%qw&e$Si?}hsJw6(_R;PIXQg1>CpF6<w^JB<8`?{r^|xW4gKcZYA!G5K>HOLoG428v7+A*H^Q8cQ"
    "5>g~3rf=i~PVlo-O3ZjP9h2fowk#XE(NG2vth+MmuSvL(io)(@;II5ydxkrg)#v+xkhM>)-"
    "{qYM0+cDK<(8hm>bXD5=%M`gCeZJ!c*<F*%l8zU_O`B+9u&1COSbVNV21tbf;}>?u84u5Y`qMxOSTYuhLb>he3$hg<}!qqO~d<it"
    "j|jhbH+c|(^;QlB9n2%j<0&fMU+5h#WAdhAiim)$h^mV$P{k6ffT$JvnDfa!9^71?^N<nyxta0JV}$=>Zn7g$QgZnXm}F5Ck~W*Q"
    "b5*hfS3F!N87kbZ>;^cI8Y~IBH7h%ek=T&S_x5XCXWlRpu~`d?s21W5WMZA)qF|gW3rj7P_u?TWG73LuP>J^QzVb~GjZM3-"
    "L{8iYuS_A+z_B<a$OQ1OpZo@%P1V>uK@)zBrnG-J)bt9#R}N!C|T;!jYZGU4~Bv-*}-"
    "7VEzfl%XYj}&w{<Y&o`f%^!K9NOf85tKQ|pR}tz#kqyzI(#bmVYF<MK^7U)q;Qn9DQlDkPa`UazCoqAa?j-"
    "biK$%UYEM|32xkzo+d6yxNXW^LUc3oBXKoms#_O!d#-"
    "Y$RMb{jL2~4VXld9k4ReTNw(O_#DU%6ohvJ{m#Wcr&`!T=*iqhNY_1^3N%(`~f%UUJBrQ_un~Zg4F8xT3(El9+m`qF#mML?^IRhh"
    "Dqm4`*t*6;Krg=gSvY+cDSa?r^Kfl0mpYJ>tEQ+)6BU~l&H_N%!*?tpBTJJi?J7Y6bpeG3NrmL9^t&#Z3kNiEyozx9fbs*$hAyrn"
    "OEw!Q0BjRbDS-Wpc$(f>e8BMn<JIT^(BJDcX?aYN?>W!?>XXz1b^=(6D3SL)LM2^N676n)9pbgf6W)zuaWhVh79kJ`qds^HI|CT%"
    "PnOOlPRt&KDn$FL1l7u}vfOlA8z!s*P6|{2oIxM>y^MiEgb(TDFOCyc4KJB9=A@bt@FnpXuTLdx0z&jCA^=BH;8;b22LtMuPB7Dr"
    "Qz!h!aKJfniMw?7ZJcG<FmZwv$y!N5@drHXNXhw50FWVDEYZ#<)FBQDP<n`dRY*f(_UDO`VyaIiK8UU!&qp?ViWZTLWhQw%W|3(w"
    "PCqKJjL9(@by9izP=zR03HbqEi_u&%OBxWCn|6xpNz1<g;@^wkUVVaqum4Avz6l>db?)Q%<scz`$-"
    "~TqwsG9SMKAgdInbA1miDoT>Pc?ZVh(+O&!QF{)?FGpbF#MG*grdL4P;ri(M^d?KH5`<I%H~P4u}qpCC4mTEUgxfh2$gBS_-ej=G"
    "cr7PF1kns91+)WVEz8&JLBSsl3eriJVwv$);R;~1yyFnZ7W8MwZc%SHfP_XZ~1*TTMXylt?QMHi}1WV)HuH`2Gg&1eQS@CaEc_>o"
    "P)yCE}I-"
    "jV!hW&;G%(bY$zU=u_F=S`I+e;n%f+$2g%BtG}!(eu#>bFn^EM^zx0yx&o!PzfQsCrBN2IG9Z~s=fin9jqUN5PES`jn3tY2b`3?6"
    "tc#hiaQJEmL;mp15k26h{6>c`<D*UI|aSAiV0|m-h-_PKSa&2wfHwC(C85)DWA5nuvcdBOVSR|q2nlNOKP6W3XbF0H_)55gZlWl8"
    "{lIEzPu|Oo~w?{WnU?5s{^IGE;wz>LSdFf#GB5@sg2d4$^dbo=<iO|_SOTuNYr8C%}r)*A)CV$^-Pvb@Hu*@o_zxCEu-"
    "n4Ql!3CGNiJJ2U3N$b}D#XHuX}7O<r~Ff)1v=$`eYs8gqjk_%A~Qr=EuKvKWUSnN+LoS~Htunk?V|_^k`cd%H|_T#Ot#xXrh5l=2"
    "iI5!7jeoHBd}p_HH9|Az5_@1+i^pPXd1kb^4KREsCH#H$L!=kso4nlfyI5+Yvnu-`v1uzfAl-"
    "(Kjt#@j$r1#{~Qd7M;Opi^`8Ihzs(*+vXz462{d*>8Tm?$Q7vRjrbM!du4Rx`S1&#rg8D($SfUtv9mv20g!(w1&$I=^XWEGl^I?X"
    "ufkMD4;dL?o7#8$lrCblxoNX)I_I<AXWPHB?$Z&J_bj@M~V2o&>gechudG_e9k7b5Ju5+?9g;@{#92K4!wd_7gdZb-f4V0-v8z-"
    "CKuw)wE?PQ^BHr%q$5LZ%PO=gwGL2ImBwN|F$xXR&mPR8S!lh~<Zt2&NUt83kY9&(?eOfWrh#1@}*A1kIR!3Bv+?00?%{EFhA4x?"
    "_BJK?<%g)M^xj`|@kLED6i$c|3du7iebS!PxrBgmGh#ii6h4lZDM8<)3{Tx$>~Y-"
    "thDcT6x}y1@YtatD3d3E?dyuX2AslwfvuHteKgTQ}>3TcMg~s?}3<66FW?c4?VxYA?nI-@<jhgYdoUImu*e-"
    "taqJwfGmLgpU~%h;*{Il%+*~R5MqTXigCVqaAqy6tBb&O@F9rzUhjXcWXSC%vj^WJ$&Y0ZDC;VM*-"
    "n0SKN|8)V298?$4jJWb|<Qz)53jz9BWvX70|ovrdL=pP^GZXiD7~V3k?lg<JTJv)E|%1QxlD31H$LAkgql_n-G<ldU~D3Cf=y>m-"
    "^~y+y)T3Ng3Woo_#cnqIYR!Wg!|fG5^s%TQNiYT2<*i0R2~E^bFLI}4aw{r%vit+kOfhI|vrYs|H@&%!myllWg8%*$0`m029LK@J"
    "|uh+bGm|Cro8mdslu&vy%N<<-"
    "i^4aL0AzHmOnUD*q%NVQ6c^MaOcvraU%F;I_s;+j3TOU{r~v)SaXBXKrKLb}J4WtYU=BYPCUw65-"
    "?l}6j1faPIF>VrYFqeB^eQtY#&d3mbq7su)_doH>4JtIu&^>{E$9`~`W9+8f0NhT~icdE9PKTs^d-"
    "J`rK?6qWvlBnNBlv|TsxX#U%k&qufDa?DJ#a9+WQu|~8x8_b_h3OKE=ddwS!P>3sw4X9xW>$g{`G&$-rfn9k_YU;T+M#B-"
    "K7;&tXYwNIfPUTl+}zJFSAa^x;6lb}#%z?9{trQ^mevV!LB<h#%KHqs{BSygUVQR}n0t^oAk@}Y3GTqNA@5K@wA&1C_DK|OI;H`-"
    "sYhJ)AEy)hTd9(v?b*lxw!b4k5Los3eyQV8-"
    "CwzE>*WN~O6km}UG|_Yi}}k~TJzeXW>Z*3qV1ks!;l8oyEZw~lxMZM2BUO5o!x@kDSv7SwX|mVQat>jCdEsnQ#MhQHfa~Pb!^0Z0"
    "5tRIY)iFunmF2D1rBr$xjq%Yxq(qx?Zen>?gTS99%pI7l@>%F{qi&DP5+)=<(beR-"
    "{3;5!JKr~COfgCFaxn=PuYzO>MZi`oE7|SU><dpYv=8-"
    "H)4Ntozx27)S4G4QSA_BeMkKcj@y@HE4#P7%w2a2T|B~S?B6@DsNpd=M<>h}v1KUfj7YE2xob;`TEn!O{L0ueSzzlWqT5GQr@5AK"
    "tu&n~kdhGIKt)$amnVu@$o7Gcv%shb5@&S6d2;JF->?WYpf9~}mP0h%uAXRL-"
    "@TVM;$snAe#hZC6(f7O%OMcXdmkx}exE?n&}i#8kta{MJRsiqmWwm*QZFzmoF%qQ4!2U!-"
    "spGu&k9srCAF{}%T}z6BCS|REEB4noWcx(ZU26M<J4$+FAlH3HJcV}E@8rdF8AaYTuV}x<4En;-z8YMtn*aBZw-"
    "|)mq~Wjqvvim;G;g9s<T5#Xqib!-dG75P)2pPyV^duUZ#vkX3uWsFCN(8VZ-Ej>WR*qL~ctA>AMY9bgO5xb48`wKCUa}0}b?uuQq"
    "u~duYKd0bOnoAS-#1<9qBZYG8M$kmPOmco~7?u+~``->+uC3`sLfkl*hfuAkY=OC(!AxgBt-c_Go|%7~9Vh@pfB)+hNpFy1)2o*4"
    "OS(I#wKY7<26_5p;68dawDeA{r3rMZNwZ)Dix4Za93-W7kqQ(Z@{%}mh6WY?$mL=}VY#i3YcXe-1d<vvTt-"
    "$JC3gRN1Y+&yaJ$vjwf?k)n3w=y6oF4{O|(ng037?J*Rm}*;~WP0J{Y+?bKLIA$5&bwV_w~&`)e6B6hOwcs+h~T7SOe`rsO+4Qbg"
    "rvZVcZ-5^=}EJ~CA<sm@;i{W)1*sHU<6vH*9nB>jgP*2_F4NK!t6yHlf^|X{=r!-"
    "lx5QQ@rN4el{w<V=?4WciCt<>H#>Z9XMD2grGNF+^y;M$oP&gG`*u+FP^E8Tp=#)=*R*)6f<9-"
    "o1ags}w&i{LaGOV)Xkgcsw{NNV%3y8FJ#wF;(w68#3uj2n>1rVPf41Hb^Ty*)Tg`9x2TcVD5!L9Hukp%xls-nZ!doV6hNj!Bi?K&"
    "V-P;=-x;M}pzxT_|5pcyceh%V_jfVgI$Tc>$5}%-s!+!j2zZ`Cm0Y!whXUS8q;K1*w_B*GCuw>2d)ySpAYMHgLs3mlM?=*;@vV-"
    "f5^LS!%-AK@dghrp-Q4M15#VgxtZQ3Sx6PxhfvV%x$5;jY`+oj19dDZV{^;C2L8a&>SEr_c47;@kdv8*z$FL|GBa-"
    "*K$wc0V1%16q~okdd*_th}l_=&55y24c#u|bd82~)2@aY7lNj($yiT~tt{@AoV&WpL08idyW3d~>RASxkI6nwEJM@Jte_!GX3W7~"
    "=KOoy$Zc{r)%YJc)91(@kIO3EvRoE7x-kb%Tk7N@2Vq_&vCu$C{kq4)=Q8@GO#-&%B>CuLyp%wad2xU?(^018>`lh(<-"
    "7%fcnnX|x$fnGbRnZ;k6(#oLBCOzw4FQ^ou~83c<<k6WVNY|a!#_Q1n`G|P5~ZBd{O-"
    "i<tpLr6vXmM*6>BPazv$?ekz+FEkOz}i$HAWf7`S*}_q4AH22Lf<0CmiuK7dv}d6tOJx`Vz1AORqe8Y%9}wf1<}1tc$iP(psBn5W"
    "z;qY_Y@&wosGmJ=N2(+%(FCk2f88&16jswjpzvf*<}A}jF6PQ*f{MU-rA%fUd4v%hW1cj6Jh8Te|?a9n;%w**pjfVj~$iGD{|s28"
    "5|rvP&zYVd8Kmw9>3*pSrtMLKF8pJmYh`7iI!$FUwdUQar2(h&(*OexijDTm7J<>ctw@C-|T#v`1YnWji@YJ@Tz$qd&wtd-"
    "?2{ElKVfs18LDQzJmB>m_K8**ajHN@wHT`yA3XkMjoxuyY~goQB(>CvcDEKG1e}}Mx1=l#7CNbBvn10sSycC$YI>~&FovBqF&}B)"
    "6)BEpES--7Qs2)+HT8ak#q!Rx5;JOqt(1>p!ShWo$FQ`Z(yoRc9qEt(kZzL&PrlZY-cbYNb}Rlpqj-"
    "lkoQ8azfo{Hoxq_&4{_&F^9Wyl?pia%APJm*ol)}__bo<6B1a)zUFl^3shwVHtGp#p=f>eDz-whtWY53nh5q9hMBo`a?H)qi)mWm"
    "dQtYTi63?DSP6Cd>qG>_>vs8ZTf7SNbDur>Wo;{A9s8f8Ir;-8=`yI_rsz`F<dKPy-"
    "!I>Inq~)FTYmImj_B*xh3W$=R_@;rcat#+LvG=i0MsI_Yu3YsIUOIpRI$nw7^erD{+_s3L<Qa2GcS5lsyP^ZuUwfad7{|IK+FmE+"
    "no&i9)_unVws&q@0DV5r>ni@Uk7hSg<5oPUZ+B%8k!!KGPXWpCDd?6$q~eV%^TmWQvb3WRm(12$zmMFr_tWi%wB=1ELHm3Bbb5Ru"
    "i_%8hAuD(?v1|S^e6mnCgN(P6oIw&Y&G10C6ggp6dmtB5qMxPVF{Z0j3i0LjqLs+%&2&~IbIGIZ9Gr#9!gj2e9m;!-"
    ")0qV!LVvXQQ7rZ$8GS2W^_gwk*iC2>DJICHj<%o1?ta6l-"
    "QDLgbRyRucV(;g{LfZ{nux)_N>NN)(zh+ti~0%o81!t+ywELbg>zyTH^rMQ8mAufFnNJhX#?d5%<p(z#8S6z!Mchy^Y=#agAZRTL"
    "f!yl&qm5`fkWb*+*;($;`vqHDC+sfdh#N09EV@9ub8i`&8Itp_!61I+69v{FnP+mj;C)7w(?ua6dQa)E4B#n;gpYg>Ukz&nZv-"
    "`hcd_J<|nEE@prPT58jef$34hOioXor<p~WX<#*a9HJ1^y@u*Akmy(>I&TX^Hk<UE?t0Yr&6Nq@?m5Iu3kODSkZ0#34)gPhSmk}Q"
    "HUVPq&fU?z7G2M4(u&LDqAd1DBWFVsV*z*o;auWsE8!8n+G+lwKen%Yf06%!Ls6H^JS;w!NLzir+me@}_tW{8H%gjhr4+76O5TaT"
    "=s-PTId?0uG(QF}UI+?smp!`zMX{G)g7KT7?1*g~v^o-wbVGjg2*p`Z1%X<2WBmTK-td$M3SHO>OY?T*K6$1qAJHw)1Rc*a)BEZ~"
    "I_&k1_Fkpx|OKR;jT4E&FmS$ny5+JQ}PA_z=C2F6vMGrE`9I9F5f+I+XccxU14>}T)nP*tYoWc_8!s$LLR#ii>JXwBp^w6Cu4#mZ"
    "$wiIX{&J&1>Hel~%@WFNUtC<{+ODoeQ>EuSRQK;l%Q@Z1A2a;gu_n(QN$DT%)n5Zb+FXy3^mL=6BY?tK)FZk+hfkK>)<-"
    "ii^3E8y=%kNlGFFeEf{k^$J`jyeu&J}=jGfw^~%e7WLF|eM@5FkQN|D2lz_j{M0rp*yafI>lhI;6x-"
    "N#J)F*?smIZ@6y_6^zPSA`ykYal_Jzd||9)ONYO$_Y_aXo`hu4Mv?@5c(EwBXMKK0hjCuGWlIOQKKY>&)S|kpljFWKJ0i^8Dhgxi"
    "NN$-"
    "lK{nNrj1TmViqPcw>YOlCE$dIK&srm|!WBg?CZy|a2K<%O(@0Ue?7b726UZnI(KD%Qu2<AL*0@2opW%?b?*3NB#sr;#uU*34_7Gj"
    "MC}$Vq2Zyw4v|V$SZvzuMg38{T9C>#()5%y0q<Gij3Q5k<=P3z8x!X%7`lNOEH(PdOA1l1*${*S3T0!rtRBU{DqHRFGBwGMC;n;h"
    "Z7GN&{ls~;CGI_2yZjK;^jz99(li=A*lIoIt9RZO)9&ga@y>+0T2n0`>TvZlewTp|G0D8h)>A6y>e$E<lj6JMy>ierzjeBx6Z&T@"
    "6Y$vHKgYh+k{yx>$5YCPo$JNMACVXWnIyhR?NZD#8!E{rFA;;cD@p&6pj2Jq}Ax1;Vc7HlpDhp!&V3pTU2T<agwlqK2z;!sGgSs%"
    "h9{!V2HSG}t#U!Z`&7YJkY*}2QH7v3a_9yO{vktK91%G~OtuoVMcF`qI&C$C%*SdZ4^?TTt2|_+XQ@xTeHuM9;DXRp{tJAT6Z(cr"
    ")$xdcH(3nuglfLj~<6jJdyOJBNT8>J)r#39GL7wIbmR0TS>9ydf2P5VRCV0+@ZM@K|EBs>XS^MmkmQ%>kOqdXMe|QBXbIB+qavxp"
    "0#pMQ}_GS96ATw+H1jp7^5pBW0QP9+Pdq&-"
    "+)ukb==^)#iJ=S=f3)Y)7`S@{rH<!He$H)p*XZY^7NfGIQm#fxmZ{d_wK5NtWlc3`((n}85GVubaCp@4vq>X`ssgi1oLGftN)e`u"
    "93}pMjK{t^GTU47}qy5fuTrQgd+KyiJ+IC~n8qu6JiSc)0f?Jk<Hg~5_1}CbOIbXIWZD801$X7-aaz5YAlRB-"
    "|5YJVEcFDNcOoJ`felMxa?4Go;3#gVl%BB894R2&YzuLySxIgodi1+|Xe$mQSzD?BsfbY(6N-ac9M$Y^C#UK)SFS;8~Y*R1-*f-"
    ";dG_bpeC8}pl&pn>)_}Vmnzhj<Cd+hjj^<MiRe)C8AUNvX7l~oIxo*oHM_e#}et0)u=pVi>nZKt#0SF>f-"
    "*7=CRbEe9mhcup?`a~j!llPgAP1MVlT;q2Nza_i$$m-008hlt2!$4KC=KV@?!$MoZ-"
    "#1=(XK%qCahWT&E&6zpm!1QdbgW9oGq#=vs%H%;jufDmUVdwG8#a9?9soa`5}H=uv~Z1Wzhy;FLR$H&y&%s>++XDdE(M*#+TwngO"
    "H00ckI|~c+XCYcGwZ}f*VlcyYdNpea`zV-"
    "sv?BqpMZ(8)in2rC$pIv`6!I!l3Mf$diRhq5*lAh4nXO|Vt_bpEv8ZD-G0X<RpnX6P4i{X)I(-oyi>*(^18#n);w(SJbN-YuP%h3"
    "a_!vRHcbpsD`ySoQoqi|w8DB3w$o1?&9KY1Nq#2>cuyKrAZR!g;YZgZT1<cI8!>42r}|ul>7EsLo!H6FmK^au1+=YaCowej*4<+k"
    "$u9Tbn)>QlXt9g{e;;|{qt7S^3at1{d|f2Hq^(!dY?@#iY-"
    "_?)?*1M9!zalxOIWQVYEyO)%f4G1gjS5w9M7`%S~`(L(LE_0tUEoVU-{TQVPPYbZv57G+pgdL@}g^rDyM-zr)9zzTqHxz&$sniZ0"
    "SxZXV@P4f9UK9d%5CnYyW@htH&n)Y5LP0$tAeOCTM!-+7LF3w?5v!sorpv-h`gAr8N2Uh@u6A;-"
    "0DS6>TpypTcrnSxg$Va%=cx_tc{o`F!oj#@BbyLBP(+(&)~^FRj$_gT13w)cD_wH*V<(-"
    "Loq&5Yy<819cRyq0Fz1TOwJke<)NXu!`NhD@y{LZ(0R2;I|7S1Cx9=j{u1;WF&WFJ97K44k|k=Q4!hAD18I?%t-"
    "6(lG#?vSkT2i<KvYVOhFQ3#@g%+;D3%tKhfI{WsFa!I#_(gwti5IPJzCHFDZ*cmKK&{CZ(ml!;<p1FVUF3T~wi-BBzo?tlyoeLk*"
    "Bv8~gOeWKTQ}^)`6P(s}PkQfULTPXKOnVpQ9!{qyK4Y9WY9rMqd#qMp>DmhABv!z8c`#wA?dyW15U9sgWweAn%HrfFSjeg3qkk##"
    "Dx-rzsBo|IwrC*Eg0qCxEUIeRW*Z*Rvsh7ZIRqDif0eU(stW;QAF>wtJ7*{5%`)$Xli1e5hEz58DpFkKW|vGwE_5R(@ztKv1O?C|"
    "%{B$I2?%C$8ufcE2EKJ)tGwsM5MH$HUjwUgFx1#w59cyu@^*CavM8>n?TL@A#m&^}1m_x|Di%Ff^;V5GgnYEUr9o_L_UXxXG_dYv"
    ";td7cpvCuJ8(-lApJ;5nu%zjEJ){Y7DK6rW_CXSfkT2;fU>C4kp2kUsy{^qVbEg)?aZdk@ge_g%-L6-"
    "{VsCXf^ey3z}_dL0_rmOvXJT2Jeo>()AwL}W?1elTKM#v_Wsw?FzvqtApcv+D~ReG;%{uMqIX_Rp})j1h>C>XT#t;B_SFd+zz%BZ"
    "YgokV!@cWGX3xJJH>F6<2UyB2Q9RBIZ^G-NAjB0OGwTt1?055MC9XIC5i!xWM$>-"
    "aQO~({VU~gSG*FUt_Q*t@5+?)rC~SB@+{DSwtv+j7)i<-*Kpboo->(2O3x<a-sl0Z1zc-"
    "<4xN_X>wdHEsK;C{*viQ)awP}8sm=`F9*ba_D8F<lT~qX`H$=6jC)?o`My_kN~2KY8_1@j@z{DijctM$-"
    "kJRxGThcxBpdxq+jZ@+a3>$QIR>ei4Kz5F$LWhIsMvc#ZtRpx8;(PG&E8IGsT-=}DBF%P{5VXGT=+;t-"
    "Rn~&LEwfr_>TPcAbHV+F3}a$Lv3Gl)B&=UTuIgUUI|J}?8&6|xu|6FvJh=$Q{y$DKvQ_PBKl4k7R13{bLu*xT9ci@Uf0s(Y*zWq("
    "k#}1q`v-iAXi7jA6K}#X$L34{0i(~8Ea=3rgofXSh@F{pw7yr#xUnVgtC-^xw)381G;RBMBfyUPOFZ7-"
    "d;8X;;VC3MtP&T4D4#xvKS~s>1`QM5w?cbrY$&vF05)*RZDgtId$@iDHy3NbLkVrw*lyM9&L&0emd1?sR8zCu^@xu?KsD*e1GqJ?"
    "zfIv)PgWrBtHIXJAyfay5tDQwWoe=h%2_|S)Q*sOZO<iMN!m_#opwKLT38{pfA`638s5?`eRWLlf7n&601wg)spxf7Px8fHbkeqA"
    "8X?5p5C&UA9+}eVcI3Utg(vtgxFHve<5xYG;Bro4Epm8Ld>w(h0#q4k+|g@IK{8dy*?w`ncO7*vex};x=mwP9FHtVyJF(HQ{Ft-"
    "uzf-<$soPLeJ-jaPgC|c3Fw_bd%L5ok+pL*NF|pP%5WvFx`2GPkEfZ9VAOgeV`|dH-Fyi(IN?2yX*6cnDwM#S_K%TVoJDN#J6V2o"
    "I<=bLm`m~7HCK1@(``>TyEnO;WeU&s)(Hj*wsm~t?|YUxw|)WDAOWOW5q@QA5wUH?Bm50Pm`yp9ZdBvFwV0m7g*)<)%UvX8b-"
    "&;1mcn+`q)5)%{iF%P0cgFCL47u>ONXLI27Cey?r}qwj>oJ_*{}WE_sOvrU&;%Ar3)`&pF=m|dIY2nS_?ZKBQGqye`W@Ul41UF_k"
    "}7QU37M4g0&3@O-"
    "(mx6fub(BwO_QXOf7z1IqZq^!@fmbiAF4;R)A=E9fn|J*2+{PdL9%Sz_N>?!3<2t34PNnp7hE+4Hxo0p(^~N5Q>yO2UyWlw`Hl?y"
    "`bVejkMC?Brx7#Vc-sJ=TLttgtLNqQzXFLUGtK8h;(?#)%6v<5RBRq3Tc0AMV*D!%`qXm9q4NYt{p1O6O!r8B4Ko>R0+d$7~DM0I"
    "}q69>_3Gw0X4Ynx>B9j^gSa4Kk^cM_LQ{PG*YeGR?GU+c3$ZeZk$Q&Rb{VDR1~jmNclG^#ytcvE-"
    "I$^|iMbjI1U<YRUf3MN&Gf)c}9cYAovmpH&$0bWXl!`JA!LJVWaP$Jh~l*BsBajEaieAj~AWL*lhPu&p+CjEw(u?ptzNe+Ra}pz("
    "PBUYKZqpTL-_g2He``&pugZ&{By?U|_~p3hv0S@CZ1mYuVu!OUDk?NT(Y6H$jXbSCg>1Y{DSL84z`+p}F<+_SV+S7~NG+%nxA6T$"
    "CEoYN6?Y9}PT%3Hm~2p$-"
    "e`D!}~Ox8$`wmiV?yEIYo)j?<XhLW}CO~Ckj&ZhlHc8_KI^Ni|L4;BtBUG~wQJ;?Z1a_19l`xohz=U1Ht3He3Y`d-"
    "NxUDijN8u1JZ{yY{3E)(HX+KOiaCX@`Mk__{j<L^4V4H>j)7623Kofw8SM(re%w-"
    ";$wEr;KFb~k0*S|eZc<vOAhGHmxF|Lf{_ReB5}J&P2V`#mg=40j}By<3-"
    "ho?wtrTFw@8xdnM900^I1c$f&py)J(2(__heB)(7ccr@G!bi%C4fxB!i<2_tB0g3L+huv7%_j6YO-"
    "8E_;a^RPf4C`eAO)l{_*AfBjGlSnG`JUnWZW&WlbrspPj?X@^tBz!EXAR@#YM-"
    "+9$^CXkIJFLhH3=~cX=INRIB!%%F(9;RplwTZ1hk@Id}r*K1MN+sjD^oMP=q^{ugNBv0B)*{C?K#W`fs;wQ$N~O42d!k!U<0cE3$"
    "`mZ|qTGERMFCT8b}uRsQua=2J?XP>GS63VKIN6;-"
    "}(rX_QXOuji|<GU!<s52+`oy|n6Fzzg!GkJL?h*J4SZU1|kkBPO)J}5FzAN$A2HSCU|ktu3gt<a?NcsTYO%?^G6lZ0P-"
    "ce0LF_D;UY@0RJ+j@Ys7$pVWZW{gImy0citoi%o-"
    "gnN*6e>8h1ir+m7<!75oEN6}8K?UCt;JO_{T>kHCrl!U#wzSEI?crfZ(FE4nn!Vo<fdO`e-"
    "9%bCh#vHcJ<n#wO6Oo$h>pSDy=K=j002C33()huzvtr#9Kq*OGD82i6|-"
    "NrI@&OO)xO8A{d>%=HX^^33U}}O2<7yMCd;;d>!6<!BJ48J9!!qpk-STT7ESg{<(^yso&;mkj_1bJcFr#`1GRHvi54_n;M-X-"
    "fr?zA7aEI~8i%#&P65j{)NijKuk1;LJgsvg1qdTTgn(xs4>k6SLZhy-URo+q_e9VYZm8yrZ({8{nOk2@$`s;&8W+wUG-mQOAINn~"
    "D8dq;UEEDi>(tK>Ax{$!Ywwl;7^IkH9@eP!u<P;C+Rz@pRm`5Ps`-"
    "7quF|cq{*(2`&e~dTa1p#+=ayvoF2blUU9nFhiu+M<ywMA^iKND!@T_-0+%BG`BL__o8tXMZC7YfHfk-"
    "4&+{zrZBY%?n#Z>&C|M|~<{LlaR@BgDIEWTWlcuJ>FzkX-PgqGTsw;Aw#WLNRr6Z<2I$1!|!rceKS2kT`cIp_{DIa{M$<8V!gF4>"
    "5Qu8Mu#CesAFTrHzsEo68YS9`Wk$E2KO+N(%9<3O(k17!ylG|$r8(Op@ZF%8FR5fDRCDb}zUBc7<F+R!wJfRA2=3mdeT$pHB+C7Y"
    "=jFZ+?NsjD(I{doqdtYn%@M*Z#<!0Oo1kA(f01P0tw$j~>-"
    "iY?mg{=T*~*}i6_fo0VALy)7TE}Hg}r;ktNjO#ruNA;Lni8UM!Pf5|2qLQSG;Eb+EJg46J1Z1SeEiHQTSFdP?ucaM~TVylBHt$v;"
    "E_<ahXR&tI#v8^j)MuK^;{Fh2n1UEu(+~G}<R}ce!)NI;#qNn=HBt0q$D<dsi{QnE^qQ4|PUB5X&R#|`C4b%n)f@d>UADiE@ufEU"
    "BT}%FA4xYf?>i^RCYNRW8IG}x)3p3{qg!0k*K7sWJHzcg8imdbp$CW-"
    ";{1prG3lA8nmc!87*d}RsV`#wy}0<|M_QkXOU!)IW&4IPT5HxV(L$V8qot*&LoD$#Geh^i;Ay2Tjm5q>=8=cv$%^Y~{;Q2NXM~}#"
    "!@FjrP|9zTwU4fzfA@C~c98O_lNx?M$qz}iz!68}miy$)sY$8YpX+5hOpR_HkZ+3pQ2j*Rw|B<19g_;l2-"
    "oG`Jov3f5%9PD(4D_mI8zN0`W|x=TwY<N+pi6yj0}wmkH#qFtpM=f=e2ZP!VJ}1I0lS2)W~}L=l}iZ|NO`Q{vC|}qjt4Yo}x^9Ob"
    "pu~2uY?oSUS0_V%Cy<sE{kody-"
    "E{qEl?{r4d*oo(K!BoDF6{8y9P4a<is!5GA*Lom0oykOtvQj8cEE1c=UkK{95wdwFMWJ63M}y_-gU1^OMhe4>h}=B!*j+OICWJ~@"
    "e}@?L22$%Lc&RgQ}_T)d)p?=<(=>XjC@EcFjpVM>PaXRf6DJj$#yYuxdaRRIoPWQRGJEFM?itqsrDq|pI59KEntsYWecK)P(l0`l"
    "yj+$T$(&ERBhNY-XGKn>*`EV&gC$`Mv5R_&w=oM(8?i5DV?^A9Nu4S9lvD!;lJn=zD@`kDY4fny<K978SEu8Itx<xurrf7((R5%l"
    "}>X=CRqRe1x~u3x>U>hrKk_VL6VNpyFc-sRTIi-"
    "s3jDWtWx6RacLi}dZN`P3b6SEktZAvLFSObbr6Ehw_d?*NImJ?E_feJk9rq(#ETFZB+dHrAR%TC}ye4Zk~*50ymZ`f9&dY@IvEag"
    "e_Q#@wh|vt=rgXTcew19wDkC632wy~00>YNzEMS0W3yP&V@uznPE)6uB&=Cg5>0z%__^24!hT<jJ=NRnAqx$a^CkEGy=%N1ybKl~"
    "B6(A&c07QS7Aa&F~sO!*1e8MOKe~X?W5NSTCRRB8!Y=-"
    "%r?Q$G4JYfS_kOQa3EHGI(xinUNo6jP*u9WRcsZV`3jF!aQ{YufCGH_(Yq1v~8GHFNQ<kU+0@8;yrJ?Pi&ud6h`Zgs%=K{=xO?ka"
    "KwMVdwFZfX!bo#gg&sfLXm3H4$<Qj*PgB^?#DI#Py67*J~F2iZP3FSl1JXBZ>R~I3<`9SSWUvA#j-1h+)w;eSJR1OkTA+!-"
    "^)aUms<d*clB-T8<AiNjWv?I-"
    "a6m~Q>ZD4iZ|xMJu|KbYnk=>8dJ}vEM)nq0sAfi?Z}>(Q?%)6cQ8|Dm8tcDrvl(K#SSg1<`r@AX}6@nuAQMtA;_TkuyKAfDNzyF!"
    "IoOfApB0DL`A<eFKRXE;xxM>BIpE<;2q;#B`I&G>#NH^AU8i4d8N%?6oD&i%+<*nXLP_^vBZ=gv~mr(RsGtMh738em7wd7jxD2yu"
    "~PgCZ>7E>fw#NAhfhgP8p1pI?RuPF7AWuQy^RDl1uVh1^bN}HwcLv@@H6-"
    ";bM$++wQsGY)dlalb$IF69FvKGZP0{Y_xvw$mNwRGBW0(&Q-`Rm@Y&VpM&_-pDmmnuu-"
    "9jFiyl2t*o6JocdY|+sHPL`js)Ndgz3)ys&-j4rWuKC)Q>IkHkZjzto!*&BFR3(%_o6p61sy+2$$|?I-"
    "JE&8r3AE6qJq~nj=UE+Br^LLjxxdh0fCc*svBv`*8wRMFS0m+#QHQ95WTvUg2NdD>Ky5wQ>6;;kChZ1?W#z0XH*43_zEIq16nf)Y"
    "Nhx@|)k^bk4VBH`{B~y<#)3O7FeYw!KB5``9ua+tr+<$XDb*4_uMb^<>Znfr=(zp{=)b!8S<o&hX`Tzl=K1^@;yMkiC64E||ubIO"
    "ta+^ckrg-D7siQnll3-"
    "ppInjL*~|dj!A<J&tOkH6TQ<wo_&b+nA(cqWgV5My}QQ9}zuB(fhKw*T`*AJCag2Nj!;dRg0~eD;3^gi{PCrfwJRIp<qUU0eD{xU"
    "Lyh_@x=J(Dc<Hpq4}{{Oi5fopx0W2-QpbMlE!U*!rCDCAT<s$P*a_3k-"
    "8jr5{o+uST(WxrVs}D%4C(G=tyXG*{agE@Qx=(ww|3dm2X9|>2J#Y>G_nCx+j3Aef$}RAdB(%*Yaz53VfFgyc)ARvK&sVLeJ7>-"
    "6wEAIUsNLTU`Npv&f3KK3CU-U)ri$W$!iT%8F9XYck`S5gi-FoxKC7eN7&&N*+{7-"
    "mO=3>5$YsCSVP++6xpO>HDB@_%?BHU>El0^7v^00~nXiK{@+Z?0-"
    "%3zszP9DdLL6AWrjYw@{p^8kzKD1v<@`^3HQxcLXb@45c+2h60sk@U+V!zd9N&)BY<dqHlfm^}xN6WCtBnkHb-"
    "G+!u|J3t^ue08gU7<~&>00_Q`IqffDOFHM%Us@sv%U}^2I`@p{7`~%)lzPeX6iL|^O`3RI_Z)jaGT9-"
    "S~3p`fupR70Gn|q3vcBYD+y9(y=`_cyO=~A+a_7LbKw#JJ`(B!<mN&&uRp_)twt?UGaz0vs5c{0}z&RGzW|9-"
    "FUAbKitg~|hKLrMbOY)xLQ%-"
    "YR^pIZc*R4nh&am1VI0tf%yx@=%Ba6frej<!X|&rc@fS}ySRgCRrQ1}(FEsAG~f<Yu$P_nPb@b~qaWo{Y#nGcA2gof2qN&n*PVEz"
    "jt8d@uJ{tTrX9(lz;JwZ)7&KKy64>Q;1UKlWIFJp=V8g5X*!c&_fA)rIlqQsi%Mw`B4RE$vU!G`8xg%i<+x>2J>r0oI;~R;sU&k?"
    "lEMJwL#$mG-<vw|7SF^6_V7KO14w%7xC_asg7~H6aPj`9c@B*Km|xGvD9i`B=GlCSD|previEBC5XGm^BA$Jm!0wKe+-"
    "q8zHB^UHqfZLgmP+r$a3htgb}awKAU_B_1U|C`I`;^_%c3`hWy@^|4@D@<=GzI{q@<oW{+?HLNdp1FO$cwu0m9;I<Ivo`8!R#cEi"
    "ucnA)rR=rwlqeefCm$T7Y&iE_lSg5O-L9AC2%6Pb!B5N5FQ-TWl9+9Z&Kbs~dlyr&~)g@0xk16K28d=?y@veGuZx)%cy-"
    "s9XzK9urVl2m6W9LFk_aG|lPPRC{YhQQD3=KOoqt=<NAA2l2zpJ)PaxY-pNexZW+B^~tc*ajX<XI(COnvadCKEGlFH^)3s^!*Wm*"
    "gv%l9)9<9o~@IiTZ)47BjvPAirPNLBKq-w9Sp4;W-"
    "oOS|XBIm_6izbe45M`~EGM+QMt~&5W9n2()vYwZNk28kiRA*?JQf^&tu_LBY+6gqVmGY5N4GZ2zX@r0-`eLI17I%-"
    "0%_jilExfH=O6*h$FZyO*KVPB|z5ZZ1h9nO|w4F5CCs)4gXNTAj$RES-"
    "pS=`2qma(JF4B-E2V&Kyxx=V%X1(?NH`^jz+ac{R0pdi#D1+lNLic~N0kowMl^M7#o{fVnl&NjzuJ?9vPJxY)rY-pun=>O^Xs2w>"
    "kJ*SBs;_c#$P7kYoOCLNkpL)%sT6|2^R*og}6HZT2@M)Ar<<8#+<G054`tA`rP^?T`HSw*A+Oc(fdQ&`dQ9b@5G0`zlM!e+F9mrv"
    "s`UwpISsFB=}QV6cbg18jiS-"
    "B;3otd`vh{nD)Eb(0lz8KhlR2t@LZ}Ho0FMRnDid0rzMXNMI@tas({P5F)kr#s4x7)ZK5l1XV5aE4pC)CKu&R}TVzsi{<=|k7rsd"
    "c{m3%HrSZ={9pcTRN$%-"
    "SM7`=sBQ3K+AFKxa$X+MRA18hP{=!o#!lLndmvXZpXeZGNB8JK?{nh+FMj=UlKAzh5ch#*(|qBujF^{?&oCCkY*aV6g><jz8u)C3"
    "T02rq?!S(-ZZJXi(tqRrM+A>(C|tb=GT@t3WfnPcYph<E>FONkVz$w?GHymD|HGPN;L+P)gDL`)`-"
    "z`K~oRSaA{3_^s~<dLSq6#2xI#C11IAC<=c+%ThzQ2n;}4si*he?%1!R*s)Y;`5Q_OVX`|_9RPQie^(gQS2BfL{_J+R!d>Idm<&0"
    "_PX#iQ>yZ3KZe|H}d>9?mDvDt#TjfxU>G>d>l-"
    "+W6Kq&c*qZ&c!U3#PqE}|H<^PI@%^B(V|x!K(hp@CSQk!Lnl#uFD8weKzWye}@17PxK4*^9I#C0!!b#7(sO*X65GrkGhL#8*kphq"
    ")y?xpE1#3VDbPIo4W1dF-ErhSJ#8UF2gRy9{DrG7TL#4uXJ>%Uc7pMN{A-"
    "zdDNUO8xW3m427Ll7b0ZKtQaeMk_ldstd(A??i1^?oS4K@T6oxU9w~n1TN_ZtpIUdkoBKg@?gHgFwY|2#Zgd=HP^n<<T{6ipZ?^@"
    "y1=Hgs@_2Ck}U-"
    "35e=F7h=18Q(v8_>e>)jdm2{CUO0y{;mZQhLTg<RzC9}j>@>D`oi)x7$e^*Y^bGgHHDArd{+HGF@72JC5+NZR{fqwIdI0(^tz<}5"
    "jLE>A6=9e)Z!fecFyWAbcy3VL0r{_^~wlo-ba2D-P2Q#FR37FY@qP1%P)j9=ghJ8wea^E}DYkRl3aIJmOH|{cQB-"
    "a=>5q)x<#KDN_*^?utE*TM}{Z%onEM%=Wa!O`@n4xLw(4#RtBI?|0qN>~{C$-"
    "5{X5>zD*ARzGZC$3_Jq>Ltp{c?#HzGGMJ(k4ZQ{Uh>{Vb8z@AodIsp}Q@>^P#(&);y$W<nU>lQfaDK9W3~CbQu_2_mJqsQXFNYP#"
    "=qlyA!0c9#un&j6Ah_$paV!3DewSSSgeuQVE(Z!v!g!nIv16tcgQ;5h%d2FX4MAi(ms9bgGBP*Iy*Z4;6mG`+D)0@7>lLWtkf+EO"
    "voXHW_{N-"
    "JtLuc)3KrQ!_^C3gay?;U?HQOmHW?Twt*!l<4JA$Ma!CtmsbTMD=4G}N#UJsO5TbkQ&|rnJW#yPgRMy}syaMO`=@zVhD1sIj$EoR"
    "LT8s%##iNQs3#)b#5P4CzGb^GF|)4tE3>Eiq$!FkwjV{->*?dqJ~y!<{7{m%Rs~HB~_%BmHT05RjX$s!+>4!RAcZUD1odT#q?-"
    "(K>Wms?B;vJ)1z<irHsolDPZ)2tvH>>f4q$pDE8GH@JqNc}l%V5V?1(9h63UTy-V`HDxz^p-"
    "p9(REbo3u}9ksetCRIGV>yPl_L?3cQH$-L$?H5pYPG^4^snk+@Si37xhNB0#_d+ZXJ$xzOcr(HD-?@T=+)7u|y4vH-"
    "<foPAH&Crw^HsX3}0kbDA>#a`y!z@CH8HANW;aw5$*f3|m9>f1_>P%VprG1d*U#F)|sFglb*GV5{}+jW%%5!}o77oAr_DBYzTB6%"
    "<&>CGKn;EoQNU>iMO(F*powdr<u#ic|}F{GoI$Wnk3yW?0>CGkvyG@=R{ROh>q$?{G`bFDJ>3;8v5e&H&vY4395mZrEB!bLaCE5w"
    "hQ%>=7`2=n+P*HC{!bW`e!5$r?#807|%Tou%y#H|2ZZ1OU}G-6R5i&HlabTUW0=E1kVu#6}K6<9LqDSI1|6*_*!kr6ICsk8R#&<-"
    "Nx{-"
    "OUBPxjxR!dTXv$V&rpwBznEtSZ9+D5^<*$Hc!HTj04Z+Nc%qqSkA|~HF(J&2M{zv=Dn^4(hPn9%*z>qMu(hp)l=A&F_w^y)ey7y;"
    "^kH&saHVOhL0+duQecv7KJ&e=(pTu`Dpw2soni<wtdzm7Xlv#2<NbsYq!+?`IP`h(GKS}vL8=$9Mh1f`=l}kKTjrg1mWJE7`Nf?="
    "w{kFdobgQ2?^gcHj50S0PCCQ3}&vnlPjrz1_gMS&%pL#{ywtGx(0A|)t`B)G_PW1MJFg6Qc2vTMg}XHbBYt(sFI-"
    "Ec5`YRTG<|(ZrEm);Et%;|1a4$sjZjK0qa%c#^RGC&z{~bD-YAb4qP^OQ&%MK{Ak$z5>0Ese=p8t4>T9ZXZ_gD0ns9x%*IHk<Du#"
    "ixygFc&jQ{1T!L3Hh8JP=_Kp~yZT)8RJ|<I0g`r|~mH3!C5ps<)AgZHGzG~!8+zdVWd$k&T3>Kh~V{f*d@B5OB`0{uEWr6YE(0l2"
    "kzvPSGBk33t4Rq^4zxb^e=fuT!Oq|lHZ*x560IV8H4HUM<Ix$BuzmgQ*xL$jt6w&*Y61pv$!aHb>O-lfXs{Vq5Bf3x%c1-"
    "OXvruAb-(wo^C~7u=)Jx}dCIr*VZnAieMST~?rq+b)sTPF1i!MNh_ST_RH-"
    "6!j(CzNip_Kz_79Y}%d{$`Mxa=~oDY}u=et(fE%k|8(6|mOJt<!3z)ER>OmY*>_WYhbVLw?L9?D^{6*PylUggU%)u6s^7La&-"
    "JKgmiGW_7Nkcd!HzL`<;rNshwAyc1AkKiP`S1dV2MvJY!<BQK3;JDtI>Qe*~dkZfEU?4>Jvs9F7Adje}kYD@Yn&`41RfkGEyVYY8"
    "=%24wiG;eFx<IfB-m3OJWd4u~e-zYnw2*uG2>Pa>P`MJ21?}$_ta5$UY^@8NcQvZlJ<L#)S0;+-&uglpcd-s#MA6vu-I-cNd6))G"
    "syqmpPhvQ5Nw0p*i2Da$;KKSqkD0O5sp6+w(U!VesnHTa~y65am;Ns6DZj{{5=!-"
    "8`N!EU_khO%T+VoueIS_<1L<bVPD&rz+Np?$wCS7=hoI!M;ps$suL3V=epS|<To#uEb2Fyf>H75;6?Rk?<bTwj88!Rx1p4-"
    "dT?bMad&LjW4le3Jlm2}t83a;<%SoRzbXBQ}$@gr?J<==TXta>bxaommx8rOD_Js*%6^*QkCA7`lJoBFaesvG#Kox*L?Tz@f*aiE"
    "&&wa)yFt<5#_mZNsC?^jMsE6R7BjP!UJFMG#z{7Jlyq8Hv6_BzUQjk;iuNx3~fLa;qLBr5bmZKJ-gtndYpw-"
    "??u=Wa`^p!)KkT2Vh?sGjGy=yz!7`3zmQrPH~dM1|T5hQ!=`q>ksN5xLg-"
    "k9{~xXtxP)>Js%#d>>|s^qUeV?pe}?SbcUXso8q)PbL>@=PMXr8?x)Zo~{&^+#yRGrMV09Z#ox64{}?*b(HPWH~`U|fD?<v{kvz@"
    "oaC)yCJL#M-co-0bX;eWRM&V9253R5eT1mWxOR1#HltCvTnrICCcms0?mP3SW8f*jCW-"
    "`qv4DmY=y8qTTtXv$)GTGfjde4jDU^_#(s9`$JG{mV!@x^ZM<+F~>*Uyp3*Qpy^^94_M?ep6zB5Pt3sz+LX3#h%^X9jE6SO)xD~2"
    "=LI{75OTfHGZXcn+#_)pU4s?-"
    "EDI+7ym@|)Zs1`OpqPw_S(XVCo0*(Duo2Y>MOqHGuyENzpBGnYv}%%aXhI%;Wg1zx<h+73YE;KlFbF>iFdA)5BH1~SPM{S~%*6u0"
    "fBz?!(w<Xrt`Rb4$AnVum9uH!bRGbbd%t7wVh5<s8{nDVgR&8ISLAV1igCQ{|Bz0U<gH5>9Plx^5L6S0<h8ULM6!hi>ulZrGN&@("
    "Om==3A_A#|n8n=D}>MxHzbu+LnE%?CMwj5^Pyn(;_F?Y*_?M$oY_@fI-Lxz3I%%o$gzUb&hyYbz_@)0k<Wi0eBN*L`ikEVRRwZPc"
    "B`St%8aubwF4-"
    "UdKY*P2fu1lXF4gGlTp=y;slCB?Xstl&Rg_fjL;su*#bre14jC&CgF!8}o9)Q!dy5zf5Ut233^NDM1)sDriAcMP8-"
    "iWQKkh0t5#blH-MaG5zZx-qgChD!UqE|vy+=L4m(MPr^L#$*u#M{_IU5v3y0=~PjGIq+m*2&j9um4mpI#$b&Gyvo=+JQI^t37GvC"
    "S7*%mDD$P#5P2lf++>lQ{llEOQV5^69zO{~B$_g3W!xG&xUW3ML}vCmk(6d$7Q54l2$Lg)oRD~umT)W*mhBl5#5xf&86Lz$AfmDd"
    "2x_<`#mZ*bR~}kiADnLEw3Z!WaB8I97zBE^dB*VS;Q1p5%Dzra0I`1Q4oKZFw2vgoHfqaBh`?(8Ric9mpZ#444$=K1RzH%(3C)nz"
    "hnhyOB5wD0WhZQ*>5py%VMX?f?+4|wBe^p(-"
    "kNdgaogTFTAtK+9pFXVLrS?8i_J6t()C$C9VOO}zi`hSpuVCS;jF=MkU>xoR9J@)yfe$*b*i{Cl-"
    "W0<z56*QAy*uV%8L8GE5(Zg6ni3W`yv^{;<G0nstbF{*XvCt`<=|k(t~>v?`*{GYw)C3X%3))_IDo`rF6Sf+{two2`Z6_SHSM!%="
    "g11#41l3A>cEObtxdv{N=;@MYRg)Jt~_?j^O3xKG_nxR+mR69tjsX8kxSvMV>S;i-l3H5sD*Yor#;At??CH^Q&fD-upv`!D$$gT%"
    "|9Cp@~OMn!s^&<3msGiw#w5Sbl`<W-"
    "(?DEm2Q;WR|qV6vNNr&x<M$sBKosPd)~4_jAU;G+^AN)Cun(2$JVX{&T!N`xgjDo26PX)aaWu<7UZtdkn{udP&H<W*S^dX!x-"
    "l{{+YriM=NWCtR6a0sYcF5}PvbF%GVSZ_n$#nW0Ta`yKN8sk!TbO4Xy^ItHu0@u*T9aS07>&1=-"
    "dY6N@U9?lA&Fdzk^SUTM6ykmYqu&^&LZzdY3O@wjD(E!X`5HI1;TCdf0N9To2xxEu>XwIO}BAIFJ7LNKJF5{ass>i=i#tnaFM)Ef"
    "*5@y-Aw}dVmH`ePGT6*3fJ~GSrI|J9dmzzgoJ50QrMk!jD%kGF9r-"
    "2NC2)>0sG%9KT;P1<p+vK~XjA;cFR;I|0X9gv3&oH*S6TfO7@{(9m^#fcsHTF$_L_|K!r>z6GC(O4@6-2Ml$-"
    "bHYN0(@%**_DR#1lLQ5q*7wX091FC~F%x61fjtt}O0Po37$<O~v;zDLUrP1bESNmqt0UpCN@yo9;ZdHAsPg<SyROe%&5+RtklS4!"
    "v?d!V2?6y$vwHmb3HzeW?1{QW;R{tDRo{uaRJ_6$l(V+0~4G2|C_^l?&*u^l9x$1SmlT-^)ZlGW!A6{TjCsa6Q3|Zn!-"
    "0#<q_7jTa1T=x(aX@2fGmT<^f3nhxiAH<I=*igK=LZPY}>7S5o?K|~IJMCnXG&{(1+84L<Y*#jdU6LhyF{ii0g0L2CrP!FM@%@il"
    "K9~<{G*@V=U1O+4mORp2LDD;m@EN>QBv*p-"
    "ht8gulgtqArkX7BwUAi^(!a!n&%a|O^E%mh$*XUr9dXVp)43$kNyQT5sWhh1!`E=mx<HtZK33j$EWKZBMMp#KC=w2gmuPk=2(Q!)"
    "UVk8D@ZJ|W0MEMr#U+>R|vTrv8`JH8BG+Ls4q<G3gp|fxv>o+dmft+5!^rn@E7b*=^I0k2tH#-"
    ")J(Ae}aCY%P{uYvyp#@G{1fPrN|RKSr;weo(=IhVu$6V+o9=uKq2=&?-"
    "KqpBTU<0=rXQLmGhXA5N{!|EB6m`uL4fNRqYeNJKo*ElY#`M9!RvR7WtiA~<0gd(b4U0;*BW66^0U~~~2iU#*b?0829bj1PeNzb6"
    "J`Xh2VXU3b5mdWrm;sRiN1E*W&M;Ebsg_3$7rTY;oeF~8?J*r}U3+(ldXw-K!7<i?qcF2aYYp+Bo`$f*-k50mJ-"
    "d&5kqu6%b*Fw6yY6$?Z9S!pCl5iXPs<&?~55<{qkaava?6Cnz3eYUOiQ6Ngin`5Xm_?Y(cBW6qF0r^hy5_Xl(}4mRAmtKnoo5Mft"
    "U4EzU1#WCsKQ_mb%uO|bXC>vT;cS+rZ-rvJ!yMHpxRc^f)1}xT(mK^+x8skS0YO(Iey2vW4y#2K8%}!fD&V}7V-"
    "w@@cfj61prj%tTx2odkrVD8}DunrM5Ar+_`TE8A*}oLy*Vnjy0Hwvpc433s>d|vTI~|2eNQrCqC4nXHgE{_z0S$yK~tMjwLTMm(P"
    "H73*2^%8)=&(OBJ5n^5O`T5S3g1iZfWXQcMFiybjsxypM#4!!MzW5yI3L%;tE#rB__)8}gWJal<M1i<{u6-"
    "@O%3ihbk|+3Xn@o6C1GDuN;g3DdV@LP-"
    "u|`YZM&A#5e**wYs&?^K&DCVgW7z#lymyE&2I0?PJ)hx@2~l0q}JVx4&2cN@yZ?qt0uPAaVrerIdaUYKO8Oe4{!dV<h!-"
    "k8Q>aPPDkYm+US&^624YRwQcO};WozMEI(J$|4R?;4n3FK|t-"
    "l>K){lrKg(PbMb&J+9t8%IPa45uU*oX#9VO&?PuqesYVszFKSQWmQbD)XrnH6wRols-xlQ?qk{=Ipqrct^GrZ6I?1Xq4^~aL!TTi"
    "N!f7BguzX1-"
    "I%7EKO!7qCvRjIIltgA^BjpF{KvIKF#TTwd8^Ro?(7byti7%2`EgWjiPOZwpIoH|ZuwR2Fhws#`O&PhWc$F6@4O#vaz$CfU*ww}O"
    "(Y3rGQrhK=Q2V>5eybFs5bpE>9*L7Jw%?-"
    "nsh>@f#SG^)&_&w^wK}BNz&`G+zIe|pAMrxcT)fKkQ~l>M_I38D?!cqoJ7Wc(fFQNor1Zj`h%$@D!(-"
    "PO$vVG8Ms<WyRHfX^`#=O(Oh)#y$3b07s|Br!uFWel1QLOZ|b8jMu4o5!QPdNGyO0wzmDb(+h=7af+0;z)f;UImk#{s0ZU10=Vdv"
    "xyv1K>bcFKFHD+C1f@}Xj`x-"
    "nXL{_wQdwf^E3h($nh%sNCO~o;UMtER040B7nntpGAHa(Vp__YX_Yr04_4OacVN$zw38ZMzN1GV#0BDGxiuw;;TUZ2OsdOA!Ob!F"
    "5g36^fBQx-"
    "!?+A8IXgLX&S0tWg%<3Cph+YRZnJL_xPDr+%H>oFVu8j6e4<~8wfr01)Sf#sQuvWD%b7|@1a>WL~HbeN1L^k*hfS&QBIbo3xb!(C"
    "s!drpVs7q@%?XZz#~)su73)Q}~LMjJ*_-#R=QeSQz9!uk4CSxDV!%E=8zEo*@fT`gUxWe$ngVVtMU+&iW6iR2<CG-"
    "{yQiEhDK`D~~J%1|W@<M%^?E+mq<WSBQyt7BeRcw;GS<MIND19|$p#thbzC)2)WU8jYQm8D?#p()t+v(p)oS3ecoLcvgup|n<xvc"
    "o)V`i99a!8ooIpc6LZLwQ)Q1o^GAUH0k7fZB5cm%n!n;qW}_Q2X^zJ(~m%(!slrBa7(f;9Ux1ljX(Pw#DZP8(Ah*38B3hZ~j@eon"
    "03mB)FN&-yQcJ61vua^!q7AYnpaC)%A=+wzhY_Ps1`9jcgh8_2mKEF*gCt&k4s^|6g7A+H2W$Txa;N2>mHR7-"
    "8Kn{3b}WA;gvhU7P@hK(xuW)Br^iyx0j0{r9YARL$K?M9UD|I(_<_z1LoA&YD$Yyd(Mcp&@OpfuQcWw!%=HS3^88_2>iO)ccm)=8"
    "&4ALYgX}oF3D`L{2u5d$mBxPLFgR6ARx#=*nov^<C3!?_7@U&R%RVUMA5TKB<1lQaN0^LYAImy6h5lV{6O&!QGYIiSlalxjF6msU"
    ">@?98qmVbMvjW+U6O1_<bRLj?Fjo1>MH0EjAjzQ`tA&8l5g`sC7gd8Zkv#{9;{~eX!hhKbPbOaCHlPtaaUnr$S}!U{5-"
    "Lz>z0mN2j=_mgq=qnM3_1QLCdWM2s+zquRACi$P9Ulg_r+)Hy}I5-Dj%9h_rOAtKBHb40C`Lle`R1)}}$dosQ&oV-N&6B<Gq-hoE"
    "cejSfndO3#w*VrpaR2kyT2&*X0v>L_7-h7PI9biUUR++?WuW)<#N3lBf+DOExw{W<zs)Kf%Nz0@2nC77x6-"
    "wxF&p4<AL|zlXg`s?t@>i`Q*pXA3dam+w6Eo;v^VEaODV^K+i6<T9YU>)8F?>FE<!IN>Vb~i1^S{(@pb64RmWDN&(O2R(w6oY!di"
    "!*S3}@OreH!Ol;tZS?uVL$aA?d9;pq8PzH8XpXTc7`Wgu2d(1VSUjoS{T7WTHF@bNF}5;4GN?7@{gFtD<@qi^UHlUfdfKlgzs3`="
    "ptnw~aWuLre9ryFcY@g^5-OC0y4XG5JV3IO9(}L!fO>QKaT6*rn}UqpN%eb|+iA4}46738IW~plADZ>4q1DL0P-"
    "FjzHCZ11r+!wKO@`xRUp)k4PTn!iHN2mT|I69M}fWxxb1x6*TIXGv3}R5H+zHKFNfAF%9aD@=ly1--"
    "Pc7n{oG0Z9PbpS_CW(v{Y6OsbfcwP^1#`Jm!yX6c|D&@}o;6;mML_B-K2=+_HjBBR-"
    "uvB+4}<a!%}BjR21XlBY3w9L=bSuINY;mp6U@z&)kW=$~sb$t{0|8yXodUGslRPR&cVRhhno1&+fOU}h&A$Ut1;DGLB=WE-Qo*di"
    "F~yVPPGyEHA*lRrBf+b@oP6d_{nuu6s`-"
    "3{XKbwJvQE%uaO1?}uJ8+tG5wcV`sn)aa>It+(f6*FfCH4r0z+6x8XmNp9pN$;MZ(f2No+@WC|X*QJ|ZKdXuZF(tpv@~U8+^JQeY"
    "huPS&{UiCISxYSM0s@(YU$&)B);p;WU33oh-"
    "FNwk3^ZI0;5O?xz7QuRbNU42`E>XBxN8^P*v12TeC}vnUfJrzdjCy8_JcgPn7j_I^pd*N{Dgw;q7^0#On0yR5cC1recuQL@Bk-gC"
    "TIl)^?DFPeYj=8I%-"
    "F=;Qh79Z)NUFj$sz^@JyorZcS_?vtxK5(?8@G@l7Xb21~v))usA9WM=;d_MZnaOKOJ?>!4;F*<LpfmL(~kHB>Q?8<FQ5F&q#(Zc9"
    "_gvV>^==g94W4vlUDBY5BR9Ui6-eF7Uv*$KNZOOf|c0O_Nk7A;z!q<*&kWd7ymzjhHdJE-x_&v+BT&t043uIo1r=kvUFkza0I#HR"
    "))UD@2lhTUVZuqmmrtdp=w)Yl`%#Qx6Iz4{%PWTX0qVX&tyMvi&N)Qz1iXU3qgQZa#k(OJ3bPW%sGGgJBz{wEVgBVqm87=i;&p6r"
    "v3l>v{YS3U4_%bD6_Nb3X2_*k?Gj+a)#AD&vlo>F;h`$O1=w~bCIf9++B0<q4J*_-"
    "1!UCAHe=q<IQz~J|VHWrPmjT5QOY797aX|f^FW46xl_4fO!03^-"
    "o;P{5;3woY1?{{Sj?X4p(XEjev}M2~kmZrXB#7~WDkm)pcYhud@>P@Agqb40?n<0Rp`_07O;bPetkE3Hqw6knGSkU64QO6Af@c*Y"
    "^Hmv59rAKO1Zc{H>QV|gJ9#(>{za9i!3r`Wo-qk`kNGS`!WGBsE4X#TojSEQHb#~@Q=(68fa<c3v}D@7Yp{wL;rZ_I=yhh)KA#^t"
    "s7qq<2o~4aqdEk94gQ`#rAGW7Od2F>r!=`C8E`L1_tY&qT3;juJu6`30QJ;HkYE;v)OZhV%AUjlqN*fI&&}}M$n2krhitc)RwvmM"
    "=Z4+4UgMDlz9ELx(>f@^Z!>6Xp-&?x#%kspox-PK-#Ly-IJ*-"
    "}*z+{T+A=2?&(?miCOZ72mzY8jEg9?)I3<z7Hi!><vf&eTytT!i0Ew(M1G*QGJ2X&T4iuxM11MGlLvgA+>b0GcxzB50zmXlbnwbA"
    "kYg|6kKCCs1Ff&i+^PC?eIx1OT!0vIajMl;)gl67(=QrzzDotS!u=`8Db1WI<kUcC%?>OOp6alhMKF-"
    "3L=t86xF;4Z_kV?sZ;f<d=y~z@pBt<QNf~8c#r%L)cP$45!56a4USrlLpdMJw7ok*G`3e|Fd(Z59n>qaSM92gWa2%SxL+G#A%^^c"
    "ll_Jzceog+oL&ick>ig&=Kudr_}4!TG{H@>YOt|jVclm-~@78GbDd#{|5;nz2_%%KYL-"
    "N|cdW;QHkV2Lo8JqQFKR80o&beto*%(OK0;AOO+7D7E0y%*`2*)}f10qzTPz_o~5na_rSH}C0B<ZbeqvT$Cwl&>Pwy_QEe6paQ7%"
    "9L|TlFoYefMX6jE=d)A)@2>Ep{khbFjFy)z*XYorEd?TfeG}M*v|xd)algCCuoMAzX~;t9!SMA2BfW~J@E`5N#G45q;^$1q-"
    "J{(%dR1?pR<Y3;XTWBs{+;#w*B7Ztk>0G+SdPI)*dRZbo>sG5aW}wl4=A4>(c0Njg>bIF<#R#UHW1q@JQgm9;d^+tgqPhBSF<z%X"
    "9ZI5^CzOe{hY|;>)YBlq{l$pcR0@A+Wpxtwlq5AJBxvU_IJQ$T-"
    "ZDU11x&AP39C`R`dq?=iBO(}@&swZ8y4xJR_u@>@GBz_22MIEU4qT7dHI5qn(2iB4^=hTna#?KDz8C2jN00@OOpGu<u+h`S@s$*?"
    "vP>oYU5wpK-RxZjnG)t!IG2#bnt`E%>m^}a(72#AC)Y#kX?W{8@-"
    ";Xc=?`@CZuWDCZct&_Z?AR|c}bwm0lz8UCYHF#Lpv8&pA0q>{CggQ|;t{y@yP}&CN876%=-"
    "c@Y6KA>)oVb?r@vsqtVG_KB}hoiJm$wcSXLM<mi4-*JOn-Wc<c>YvXFRfM5@dCTJLQ813gYh3Jzo)AGGhvV39X42)EUKwH0fcvA!"
    "SZz90ojAr%}If8;R0XOntZ6~7!;{AJKch7Qf2a+=Zu^~%qa3H7oOM=OjW+K);?Odl3bdoiVg!|J>fwiGuR`=$C;q%u+uq{9V6y82"
    "x$9VCqg+LtIAJjrb^UfJD|tfg}`@PO>$U5iGpz(nE|K38JZ%$yz2w{?^A<9<E)&wYL(E<5N?*v?wla)kgry?TK6MD1V$x<61=+o#"
    "8$^-"
    "%G)(OWIT7COV8H``<^x7be;7rN!zQzC`7~oz$@s{p?zgeBBGaCX&=vS6**5ltS!ZL=z}J>^R_588iB!0B_9nl{<?>zXq(sS=9zlb"
    ">V6b9+aRV-+|M4O#c;CuIE!^ZG69DAL~6e}Pw1uR%wQh5?`=)-"
    "pq|O^Tn7|N<I8a4%$8caVr0h}TcOk5z0oq+Q{97oT|>gaHO9=k=7Ka<<+<v3buxU~XRUp`pqic0NLTyT<Tw@CEBq(<tukw`Xme`q"
    "DimK-EL?>9P#ZgN#wc(Vtv&%6?jO&?y}pR^TK7y7S2tkMJvOG|bLf8+qZBx#pYw>drK`tzqK>T=S^_nRqqq(G$+g!+P^dDv>m!C4"
    "v9zotL(lWjvLEO9BcywsrD|aZ54fe1Y-ujMtY?{`18HY8jre!wBzWm(jQzzUtLb^))g@j+b_xW^gRl*Ga|U)-"
    "P^AqxjET({=Oz@JhFE`&wGG+41hTUcr>T7Op~vuRNwZ8u`j!J_=|tS-"
    "5&X>Y##D(xFF`*g1S410;u*H8&fSRb&bWlg1ZuXo0(AoyUZ#?J$v4?Hu1qze2Pw^>rHi!3v41@V2$;f1Ph|?4t0r%BwdA!v!Cp3="
    "gVCbr15)u+qU(V8kbP~z?Jna^KLQPe#4scqRzC&>HDJ=6T0eyUkQvVYQt+H1%P<(vk!{wX>b?RQLYS_G!IZ%7I|M0VjC+SD9)^@D"
    "i+cMj+bQ&?(rx4zUCC<_PkU6pZC$fu<vKPk9gw7F^Mf3*^%a>qg|@$q6legE>59WAIL$7E4RDtz=8rTe%FO7kByyG{rPgV*b15Jn"
    "QV_wJMY@{%36;7=w~OpoYhtSG_$k(gM4DSh+ap4@d4*P|vCt?~r*A@gvWV{Q7|b}LqLE%xWi8SR5>W(=)?i%&^EJ1?8Od6WOW?{("
    "HCRtOy;?7o{0FSu@2;V#0}|Hpg|RpI`BxealcC!<dnUoaOfnXrb>!J3*eyArgDQg`XBcS^L8n;<-"
    "zBE!C1^Dn@y9EN9Al3<WngdlIt|xYi0*0Rf=nS-80mC#5&=t}-"
    "taLbsNrU0WXFlWo5M9Hadq!0!<d`;P}X#ncmqcQHGw&xpFZ(G9UAWK@$xrJI(s2B#hldKsbiIWh9s??*<KloN2?imgrQKU=r!RIc"
    "8H}K0L#@8+YCnCw)K&C(4w=MQqq`XyCmcfACCGCJI?RnldeTo14PTbS2K6SV%)Jl8deBi8IcSpg?##?d$Lg5!dZ41HPy3N4cU5}r"
    "pXmC4<2w8axNvc^HL8R6#~9_8413o2-"
    "=@q+4G<Y(At=!?+J%`iH&HXEfZ>t4qf53b<)>QaYP!UD4KOdOjhS}klQ8oDt6B$Sm5$t7!cy%Y9_8du`~}qsvew26(VS11C#?X>A"
    "^_yZ6YH%KhSJ)gx=hM#L*oIk~Mw~Ys3RuPZ(sLMy^$=#q8<~>aqijwcBLLVA5Uk2g3CpgCS>muIJ=s??Ya~a;GtBd6}E+#~=XjG`"
    "Bh1QkrD34gaChejdZEuwnwSs=rMAW^~SqH%yOanR^7amffieI&~Mw7Y>_zyPSCUI$L<Qk8AEZ>UPKUmV<SdF)AO9ZWh_HW*ZMCR{"
    "VJuo=v^+Lr&z+Qxj?NUFhFCymTACXJ>1YNycsv6I1z0Jh&vLNrPF2$RWAcJRrINrJvWIO3uS`xLHP#9+C0$&Loo7+n`imAu+W~k2"
    "-jJ@FL9HmMTDNSYh8L9x_iEV_GLu0uYdo;}}%-Io`=D)7I*J^Y7FNn-*~&b_qM8aK=6|b{y!%8>w|AeX0?K?!h^^Zr9Z-"
    "fHq5N4(h_{7e~6h>{&_OCr8`SgPy$%D2G#nc8!gIkd{rZY9kWrbk7+cuOZ?Hm5eQ8=nqIHLfS=py7z89vORG3n8<Ivn<1uoaf^m%"
    "^W@$9$jO@6Pr<u(n&y>h_f+nsrZw%}GLv%M5Qg|J9-gq(w2e#G-I7?%<@%r<N}NmAK-OBevJ`wYbcCU)4OcGAm*|8l2qb88G=Po)"
    "5y6;FihFC|9JGMwQst{%16QTc#OOwI**nvM)SpF=_VTQ+nVl@L`Ri8=x?!@*I69fw8+NIY{p=x}W7CVr>6kDDRG$G(H@D*!x{`k~"
    "y^M)5-A6RDS820E&zkk>==?m&Gu;V-"
    "$+|gfccSMuw8O)>N1~#gJvyzm;Swt%MV%D`15T!t+h9i4fa=T={Zv@bGk(PdT7bx!P${y^p>$M6sFWGeLb|9RZ0Qe<gBxj2*KoTV"
    "!%TXzQ!-v1RLx<+@uDfw$)r2h?;3EOWVABq-6^|#oaY!rT!Jr?J|$8lJ?Q9#Wcdhv-zwFRtsftOoN+tbod9M{;A8XDRL`ffrG}1p"
    "p`G{+t@t4XXH=|XGG@hNoC3=(mHV1TBKis8RS{_^eE98y0H~?#5>>t%dvdP6Mh{o6BPSmqT6F|H)Z7MH#!_>~EE^RW#A~BM>K4@X"
    "F<k1toGsi$;9_^_wwx1>C{QV!#%JTROO~TUzG-n2O*A}?^ybpQtB5O`5saO4eJitd14@p0kf-fU2Ru~6Mtw(HPqm;l9Pf-xE1ygg"
    "T!c}Q0q<VsHr+@=QL!b*VAqAH=brIz`UuXnbfrBGBR8N|Bx!+6P!YAkeo4S4k6H3TT9Op~8O3M-"
    "I|YOciscG^a}G<UCpTwW<R9Q#bEslUO{Le}hqg5&$Wu}ab@Sea=*-"
    "uhLNiP_`eH&4DAXMz%FIO8Wy);ve+As|eZ^b8MtDDXVew9Q*tr<jA*f#iAGpFT?erou&#J3z%yY+Iq?Y$<u6;;$=vHI1zJ_0ArgL"
    "RR<zTX9_JKvkel^xBODKI_f(ad}kIp|NKqBefGu-dkt!s|{pQLNKI+s1z6zHIlK$HTSwkzJ*4-3i~S%BBLsuD-"
    "B%n^KvOyv@()5Sqx;z4{iwxyFFwKXLk^4>0Mp(x?R5?nUXV5gp+(oH-"
    "Q*0mL7WyuA9ezt<biqRF3QNJpC6$0N@PN$0GpP<0kQ0!9XZLvK^*H@1SP0yTW<hhM#r!^a;Ikr~FH5MA}hC9CRtv?t-D-HrK`*|6"
    "WLS7qCDbMlH`#Gr4+yioU@A6KtWK=5m2yV44#b+Lq+uDM1`LQ{%ZPRdfXdG)A2yW>qLy3HK19?d5kY&!80a@heCASbf7CHwg8J}_"
    "IrzTpc(o5E^u%pv52R7tFdF=V8>OAVvbaQJJxTEAtQ+U`lA|1+#3m>$MB2Rx4^ofFT9cT$1J(WD?agPsZf&v47F2=}b<`;|6S|@C"
    "KP3lb@1yKvNsJW=^ASc`G&_zq-"
    "8rYIEwl71lBm5z}w=<%}x@yATmrT6?^#+DrS;HE}q|n6|p*0286_VIcV5oMff%n*VY80FpUm6ixShZWJOaMMKqRxIe4#x60!(<|l"
    "T=tuRLUEGx(NQ)UL+fV(Q5~TMq@YJgFd<RY7kOuIY%a;iRx>ikS>pO7`NZ2j7A}?K7LwysVI09tS&*zeHL&(`=sBNN<Y*0G5RS^A"
    "<lb&~YxNycW2bg%tUMks&O{QeWEe^*Mq|Yz@T`hD$Q4WTW_~zQjUncJc+sX7KxPjCTuuCeKba5&&Gs0ZJ54F+Iu4xD3EK9*r8b=s"
    "i*xk>p9>7dc8>1DoW6l7=t~S+V#|-c8#i=`YB3ebw72zXiYKm8Ne3-"
    "kEBDS^fz|IG)W3t&^va+Fm%8UTMy6^19xUcV@rNcJ$L}>VxwhbB<e3?qr{rE@=_OOd|C%bc4^MJU<yq&Y!N2fC+wH98!y3u76z*v"
    "{Ky8DZ6R5Q=qm>o?8}D1<JRwC>QxTz@=t9md)s&@~y+Txh4w^dtJo_pi8jMK?FlB{Q;cmmj7*Cr!y4>?~g|piJdA=h08jiZf6epk"
    "n{g-2~o;TgYq-Qjj>J0dMi^^xL;xld*G18qBxRjb~NHI#=0p<~ZD&XnXs*Q}XAnF_7pkdn9kz=N?KwUwIpQU+9YhX*DiwCoems1K"
    "at%db2a*jP;J)1nRt#fMJn2IiP{3z&qpWr&M)*NFvf1C*qo5Ek1ua1wrD@k(;J%8@7;{-"
    "czIlgo)awnwD(kR@$5^1q6^bE<g@ADXyo#tR-8<UAp=yAsJ+z<CLKoBY-"
    "9w;e2``O~kNK+kSF|%8t5pxDlY+h_f!<&+&Iiao9((vqSnJY%LURtR*%p}+8E$;(#cI1R9eGao_3<ehhV$b)P|Lc0hP7!@Z26T<o"
    "Gj(-kojsQprEbMr#ckYbXu5@9A|4<?yN#=W!%&wQfA3cl%n~Y-"
    "0L#!$OS;Grc#?ZLfp$hQ0j~!)&7XqX5wNf{6P@fG?zjbIF4u**ZXSIyac&s~jW|`EL5#K2elps24(_)uq<U|8%}SUNxCR{J)-"
    "qho$y{2}h0<WCH$z0tpaw-"
    "V^(rxi?CHW$FHQYaBLUL(u3;|s^d&u?1PN>Nm0eYYQ!*rFX}5x5CLYdp=T05oP%A3|a$1Sw2&$#SEHCuXepvg!#z#yM?i^HJA8lv"
    "r_>_pkXu0z+iu23#1rPHawu4aJ<>lQkw9TCjOoLFe<;VeWkzzAHt3c<`g$L#2^4K4Z#nTqh(RX3GjY&_oCc1|Ew&Z#VQw@6qYwV@"
    "S;i^%)a5=QpMaNk|9H4&fY65$n!qk-M8n6{%T60_?8pg-qqUcWMS?l<DeJ5<o=+=Y}Z-"
    "azQg9V~36|O4?xT`5?JQT|%v1RFlAEtX62mvk4d^_g%+2AoBr-k>#%V=GK_8<xA+1UbV*K>6WL|PN8-"
    "3RjBVMfq=(<ITN+_Q0$wfN7^&}LhGrsptyfLr4C*!o9FD&r;hgxzg@neXA?jk7^ZhcOVF%GkgmYwlPT;ZMBL=BTX<u*DhX5h>mcQ"
    "Ds9|lR3LrVO@oDOfnt=YW{vg5bSc#s}(vLiY6Ud#4&jqb~LJ|AXF-^s<hZ@L26sdlBH!0;-"
    "_=EI+Xrebyu&INd&k;nd<z=hGgE!(1;YSElO*s0?skX5O1-}HYY0GVR)PNQIdzuV*UT1)jElJZew5#_okc%Pk2~Mrf4-"
    "rG}7*g=eK<YSx}rB%X>~smS`l8`UyO^9~}hXG2D`}U;xnF3=nyca)K!k>AIwAD8td4@Pw8hd?mUPa*6NLY{g)<1&b$tXT;qT2dq@"
    "kQ^+?3oZZe44;72Wn|Gem#1_#W#Y8hHnNR4QHIcghhIKOnn3B|5_0ti*i$X>xV|y*Nx&mB1cPXKA*BP&I4sv)kM^ty_S!{#2*<Z*"
    "|L!>R14JqbZw}!aZNp%AkDlLnxehnzKBO+*ZvU?g|doqTBNL7X6xJ>OHn0@Z{?CZAmG(+C@W#Vw{z2h;^tij79C82nf365RUDosg"
    "LM*=n&O@d;sbSOrmHFp(MGx&t0Bms8aD}<eP&d^<gx6q~hL+KrBY-!f2I@jp<C4)q-"
    "^cn$i=Ek81i%9I5nZ*Y0`8q<MZ`A|_&Xln|HB1c~=t1mLUU<0RHG%|9<({>!)}npKW@f&3-"
    "W`Y(UVw+q&N^R>Dw~5DSG_NHM>{2v%f$bR>!5XLio;2jX&Bs3O!nhBp%_6(cf1ypbj$?j-"
    "NoPY1cPg9P@hQysZH8W`mA6;r&Dtw8I%HEINi`M;+K>96<da)^6btyH~!AOn7qht9^)7?x9?Y@(&&;VfJ5n%gH`PkF6oyDd8sl)<"
    "7jql^%VC)q0G_$Q|n*V`XSk7F$q0<*@ifLPZw$qxVH8${?4?bAqG59V?x8woL;sXr|s!SCG$qPHn$>zM@<eCXvH$bCXSJ{9J=Z~D"
    "N6`uhJzyd>L|Nt|G1jR3-"
    "xNvHEVQQZ%1jeHI8H_2cLd7zdxZ6j$drZB_9p9%uTM3Jav4ihKh)^$tp3jMxadB%n@f{pktmxu^83tSzh}&IX^-"
    ">tkHwo?gsG??n$s{e;YlaZepaTWz|^9cYg-25!6_bc%OyotQ4Pt_w>_r3FVxG-!lY`S5-"
    "R6ajzC$!u$}HUA($LikvlKl6dq67;=g2yBf+vf*JSB8h6Dq!l8lYo-*VLgQ^dcq0-JqD63Xbw@{+C!?JJ~YNfn#(^S-"
    "ev1P`@xm##+h-"
    "@XfrAbOO_!Sn+^|C;4JjFwERrYBOcP&<i<!k9b^8ytYb5dElY+#`bv{wMf)}T8wBOw3Nn|wug&3;mZFx5AJ9p_GmbwKN@=MWB`HF"
    "7HM_n4)mNvsds@P|bFEE&IPvDGiu!$w{_jIUlneoJ6p&*xxNRkinKR;MVkoy^r^TvH}tB~0l_w>O0n%EM%d?n=XIdPE*?Pn!#q4!"
    "O7F3|HlI+P|hNb{I~E>WlF3*?*~7uc6O!lz&#S=V&#L$OZxBDD~PsM{}VaLQcw{JABb<jU6N!MCars*D*v*4nPSp<^$db!r7U_Np"
    "darf{{7))0Sw%?k1YWZsvq{_KRK=ctBinL*Q0ICYq_taY8MoZTL30;*LDX)6o^@73qOyo#uG$D$nb%34mPiUAM)DEG2@Uc^TMLb|"
    "u1b#$8jTG8*dfUzH?DCNX?W)ZolRdrq07;5$V@rFH#knQ(iE{5|*NT6P7V05Qc_aw2LlnBeP#(!#m6Kx|4(2-V?8S-"
    "7HSc9ExsqsXbcn46fiB%%zb`)LA>`<awmb7yrz0+CJ5dW3jq{0j#oaK3&ZMN%OLfYtg8J=Li=Derr$%%@c7OtOp{OGb#Xa6F|41o"
    "B9iLZR8x_PY})*`WwqfU$CFHvvDRP06U2VO#(t%}GDp&{(TXgWb2flP4HWb4x*|#1dy5vD|wtXdF;c4OAayIBn8)3}n{{w66t#G>"
    "!#+iDkjr&W=vPfv%K3(;R8{o>S{Gk6bmWPi7lh&BxpvMxqbJewsXkNUk>+_(3^|f;kQ6X^RS@!r>*t=y|czN^Y&ynqnn-"
    "7!b;Rd=X(w=gE8Q?o;sPR89)jYotEN_PT_$IPc9ZH4afAMYqB2Q0{TO>w=?_-{b59TaiRC;SCy!isdB}!Z;7B-"
    "sS_eX3B<cWgEFKRooWxP2S;FAqx?r-"
    "rcF01Jl=2HyP|hm%Z!<$k35w2ug>yhBbS>L@_QR*ud1L>3A|Cy&>3st*r^wbBsgn9S`ds=ckc6T>AY&-"
    "G$tjnmP)Az=4#Rj(k;g(L@r+pL0erDGIU!TA-"
    "2vwbuF%j&K05M3U9p$HKL|M74cY_I2j?%Ua=`%t?K^qL0ZyngpR}SyQP3ri`9*wjy77E|l5h9_?LqU}>cG(r#QsqOe9;k=Wt3b1u"
    "pIN#2-"
    "vud3@IknKf#B!;yzg5BLwax!2qqlrYTe+HB?y{ORiz^784+_Og+niW?}Ip&O|?KYI3W3fB)j(S<w*&r2~(v@EZp=kCkezYcf*J~%"
    "yrJ4!8wblAp7Iu*_=eSdX2%4iQ*+TE>rh`Uia41=i1_-"
    "SNr#`H!nMt~pi`I7>TCMW(YBV7{hkK{mg)ntQPj3!TD1SpB&~ap&eIKcFCm)XGvFMZiGU1@^kPR7hdse7UPHeXhSH--VMoiHqXHG"
    "V__F7CvdFKb8(p>SW*~5yr$+RTDf@$B{HCAV-"
    "j}Ae}_wJ)OCvuVmY&A9U3;b4;OacW4Q7?Hv8cO?eynK~W9%vcM(QB30mTHWpXy3XErrB;XBO$(aq?I1eav~w3CCd?=k@;r|I26?3"
    "`z3)fIn?ur&^5(6lS{py<Ks@;NC05kk+m&2pdt2Zw(i|XW3G2QQ#9zEy^=_dIEbI6q(T}vhA&=0)K<yeWgR6#_pApD+E7gyNLF{Q"
    "ioF_c$SJaU!`M68_fAC=t+tW#*1_OY3`vx*8Gd)SPhUM+Z#B<ez*r)Q-"
    "J`Yu4fQgo|Bytpy$r=wlRWR_B9F5U6oFt0r{BV9cDwXhcS-Oa)Vq7IMIaj9Tbv&O;+08Y%jAQc{dJ&9k~w6HmpITzf{lzdpddWus"
    "j`z)o_4@OsMC%(@8r0rMjO|-AbBHdCf?Mcxa0-"
    "bSBaRf%Fwit0$e&ntr$5ZbL3oZ*aQvdc`y#7wIJi{a7E2r7DepPj!@Gwie!G{AlSN3Mj2>D0?}<0WOCP>CbRx-"
    "?Y7xVu(Owjq>e;7>B=w#%+Yb#`yBF}Y2}MF1!KD=pG4!JBq&M-"
    "c0HKF>2>ED=HE$3D>zqlH^Q=Tw5QCSKkA(~i{>Igr}MTBa^O8i+ZTpi-sQFzi@LM(Fj44xmn(<<`85s6_0F&9d4jm=Jv`(#*B4~P"
    "xPolV33Uiz!RW}e$R^7^7m%ec4;R>3<nM4?)|{POp@I0%r>gI=4@?Cqwo^Lm@znatB@pO9w_anEa46ZGZ>Okb5ZUPj<%TCCrB@n{"
    "zqTEx<ksP4s<<;E_yX#5g>}~vhzP24RZL@!WC5f*RUh;cQ5`83N>sPZ)9*ow*Q6=|!Vy{HmI<rK>=t(tP){`!-"
    "#SATy_F112MS96ELktSlXGgcCM9d!YWZRFScgsWj@tqQ&MsVXGh-"
    "+KD$$^xidvpcF@i2Ij!B_QFG)j&jJgkN@g`3uiOMmOUM;Z+LFwwKg<9~VC5{{T6<alTf>!z@cV&dE;fgErY8h#Y^BJkj7H@GytFK"
    "8^a6)30?r82Qsfca)%v<r&bjTHS=fG4-9w-"
    "Qh{iYAjwvTk*2U?ckmVL!ZiIlT9OdTpv^q_xh7xn0}Ia6I^^wdI}q_$~g+5@^usul^4<#G*IvGR`?ISTq(FBSW#GVN3CI(lwG(+a"
    "~BXQ8=u)p8Y)3=-bc3>To<GJUDX#z=b!??5Z@=%8qdaZYRo?)mO!3Lluzmp#_>Ve(-"
    "d%^O*!VN{myAV4SamfM)q=;mD#xa8N_!^Syob%&ANPuB>9JrSwy<WW(TUjwV`&Z~K5lL$1Y-"
    "~gyGilDUwT7l1_qN)!f_Cv*yky6VhrV@MQ{=_e=?_4sUU!5WY*lrxV3*oT>C+Ja!oC!88F%yjsfhIyHTO&d2zQ!I4FU9JC$yx1OO"
    "^GvjD#_b@l3O#RwQD0`&+P`$Qm4GSJ9>T>iQx2W#8Zt5&C7^}k;t}ce(JiGK%M~orH=~AXpeaE!?UWVX#N&k>{%y%R(>0`uZ&pSJ"
    "NZd5`=<ZW-NxsxfMPkrKJEEDyehFbsUkT+e=94sjx{X!(OU``ym|jU6*j_#Nn8e=uZ!8k37Fx?F3a!IfJ~i6$2)+#W3*mrUjw>vS"
    "l&DH(kl!?)Yg0VR7^{hUTwcJPR&AU>zQ*;Wd>Tskodj`0Vptz58$swG%V$KQ-seWKFcfNeD&C%Bt?AWhjb?}A>Dx>#wvQZ_(7$<g"
    "`Rwtua6j&8pdu<ZqSN<>zi0@k4!D=!T26&@=A6;yVY$Yr4`jgyIu#C8^Qp_XY|Q9XU*YA(B~%F9`PO;F6KZ~L!L~!5>rTV))Lg9X"
    "y)Ken`xzl387pwaVkwKc<7Y2;t!<c_8mrF_&)rRQWhT7gSH<YYh#Y*@H^?>Zf=wp@nlmDXbkfRv@uE)1U2=3IYaeSCD`&aF-TLxD"
    "IDcN*3UFgn!)0ph*ye0v}+255A){Ky2d^j+{r&)Kde0&016bCBZ=+02Vf@r3|5Big>+2xv{)!i&$$Z7o$=e|w?@y|N9ZHDY<g&wK"
    "wo#C+g>d%sATKz-HnYSPfI;81c8pu1fCR&*XRaO%%ADpYou1y#&j2&1eIQ|UJSG>8j^*(G;j`D!t*xgDwmn|&YA|6(x?>6kj+j$O"
    "UjUWu*KOZr(~SPVq3ln+b3x0Mdsa6R?sw4(C^oaVeDWK_A&_4-U&BTj9JT??!!oRVU)vU^xkXgX;?+w>XOx-qR0@|D6-"
    "n@V9?K1ywNvv?KOXaeK<69Mzn225S?Mr@_OG>dqmc_PGDlx38iV>!dSayww*PHDzA%!u|=+-"
    "^Jxp+nb2LM*#*zKYGIgpUJYbP<C1|i=D~HG<oWs1Z$C61Dn{qI7pOs9H(rlUYooniBcmSKSm09%ra5NC53!<=*$YFuhUre|t!ZMQ"
    "+)ESCDjbh8-"
    "+Xdv8W$3qepgeUIy5AiHEdj0Ld}=nCzI~GMUXJqit#0osDNJjAnu>ikOX7HIDhQ#{`)jw<WC_d#>QOYDuUG#i)|Ry5LM^sGbN0Y`"
    "*oz!DN3BF*c5t<hU7pha|Z*mz4U+jhBDZ$fr&{z2j%ackk&r8;b#z!%jCOjxVlELen*T-"
    "DBSK@&zFiRu{P~SSTZ!Fvavu_iJzn46<`{d#Wv#9jwoD`(m9nnAT&Rds-uM5ogq$?NJ#a(&*!7=;kV{3H3gS4TyMGFL?@WL8~qJ("
    "l15&lFOoQ%-Z8YpwMCY^j?SP3nR1kN=^i?KWMf`Q`y&Ndvtd0>w8Y{n8b%#9-"
    "2VS$MxomZ7J^ZH{>AafKYsM)U*3HD&owpnvbCYvPP@%oyj=N)Tg*avEW>UJ?aNJdcK7>Clk1Z9eT{6=Ju*PUmZb-"
    "`Q*5Xl|HfPU=&=#^Tp7`x;tx8Bh?-iY$dMkwJuK-SgyeO_K{RN_T#0r`GPp*OhCse09%&sB2o6)MgiCs$13Y>sC;JW&x`S=@9Gow"
    "TW05)S<h*07VPY|dFMfy>NR!Tl{9M}5V`4<#Eh2kR_3`+$*m#Z<BAqiI{(u?7HR3hxyfKb(NYBH+6C-"
    "`fulZ_5j1QYgQ<Bj&x6T(LM*MZ|kuK%vh_W6UD};R8-JCtn16ZLoVIFOoQJh>?1g-<m7028ZcB~Fw-MP-"
    "+yp{UUHKfi55yYqtK?^nMMb&*%H1Y#mS;4HNq&F@`5rpsZJsRpgH{&1w`q6*9`S<4*efD&CyNrUuW+`n805Gu#-"
    "`LsoK;epKJo$4v(Jl)Ek6w7%S9b|$R@(#XQp7-"
    "oQxZyN<(KkucjOG;n?;Y|WSe8L!V63i5o=tdpdHAJV0;fT#!(F>qt_639W(1#rDRFC`J9tQxmgbr&J6NLPT*1jtt!^&klOa@3*Z}"
    "N$&b%5Izj`7XT#|ZE_*JZl`aV9M13Fg#d$;ztw633r&@@$@JR0CsA-5Oo`VvNci9Q9@Z8UkaY_^A-"
    "2sr9ys)>`$aG6yYuaS@kTzyG3j1M9RE=Y42NcQ5m8m3BcCmF1=Ml(g&abFU-"
    "!<si0s{nl>CS~X?)Xuk_7WeJu~V;>M%ovW9GYSj1}icB7P#G1il8vh*O*^ax%FWvhS?n_g1hI^|6rY5QtoR|(Of@w>mUF6{N=yCd"
    ";8t@-@N<uFYn%d`F!8=?mnvjhCYxgUCC{P`JaA~%*cNrYn&HL@R9!n<g4)T>aq_VLWtBO)e&-"
    "3c2em4^|ikC+#mW;FiUGqJ`W`&i99fEw|+w*0tg7QrtY32lj9`2g_clX)4?`cCyXrlxV?5$x?o9E1rXBQl#*r|iY=_~=prHUwkpT"
    ";FVk;=f^G+CV#@^(SUK7xPW68q9-"
    "1FrwGJO47j6~uJ6A2mzpJOI4Q<L^U)u=7tfr83odF$lTC9iLRWeg+kdAyTbo1bI*kO3v)4AU|7-"
    "vnKEVoMiQ&q4viOp*6Ds@LJJ~WLsc9)9NaF(@xmmKM%w%uIDOGkCODp(N+TvaU&u0%0((d(SlxsrY9`9Kw`%Abop*KNE-"
    "bI$roP*D}Y=^U{}Wp&h3tQom-dvCArqmbpspuIagAo@&y_((6J7K#(6CNK%do6@t2my3{i+rk&|gdKZsvAt}IYrQtzNdUO#A5{7*"
    "Qva<G3bU{L7teM;SuU>n?lG4B&TOD!Gd_x3>9KM%k+B_#a}RY`VAX2i9Er||eo_8Jb9!`8hU&%WEB(de?&E?)oGTTs^`lw9VWLVp"
    "yY`XRa0YY=^$}HMJ-6Bk-Waj%oSvZ-P1mCd4V~-jALJYkNbU7eA9)%f2A1*D4p%P{22Fx{1V+il+8yfzl9#~4_esCGukL6$Fzd-U"
    "NZ-"
    "v}2H>z<(j{$OBCsv4Wpfq&x3(w>l;?^+yinjyZ86_KSFPg^G)2G|^;Jj8=y~`MXb?z#N|8yM%iuEQWauRy$s4$H_Sy=n8qQ9V+pc"
    ")@eCbXL#k<$wy7OHW^Z{=i!7t?*+cW}0!D+r>px;t$C0{qJDp<WG?sXY2bpzQnVjWJ$!u$HsHTC~r$O^fdWxBdc*MIim{Me`XV;|"
    "U(Jn5t<ZRu-eUyRV&L!S{*HbI!D#AzgY>jxXox8Hr9{RUL$q{Dgc%)RkzfBPjTgapgpK_YJd+1KL5Q1Cpz)=)Ckz3fEsz(hpm20>"
    "Jzuvw5L^XZXj&o<68a=-eLs-"
    "(S4xn}wn&Kb84t5!J=Tl#+xLT5Z6fuO22jI@fkN7lwpxbk#X?UR#*xiarqEz|j+p`JhZ9w8<iQA12PvxqVeP(mzWsb>swa*1P<A~"
    "027oNgy$o)0+PphK_V#O|#kTs_$FDx|wvuPaW!5G$0qIXTgSy3<q@XAQu$uQT@dtW`Jl`Ls8Wpu><v`-"
    "q0fB$Fax1hH1}Wk60>MtWXc3?bWQg+7By?OLQqAa|Oi%>7J6u^HxCtR9x=ELa>~iG%G%@0m0i+Sje1)$*#yMPV8$jU$H^mf~0j&l"
    "JaN+*DUWNw%}7VMK0R)Z6`vPFZEMY?L$D)>X*5v?n1%c&cSRG$=M`HfNk`da5aBJhW_%*LUiBI}@c<4wzNiQF3HE^=g;BrGJNeBe"
    "9O=6N8{owbDIx2&iaHJXDQ;{(v8Uo)Xt(YG&+v7vFU;05XG=MqWc<2m2!+FR-Bc8qQN9ythbwYJ{gJs$lio>z;XDt62a&&jkcpjr"
    "G*_rX{Vo$joCYu+kCJXbve`S&zaCc)bZPZi)-s>3i=F-"
    "!Y7gowl$pyAFuHBxA*S)<jSRqGbmovDKoWrq`@3*hp&ZShbUc#qV2!h3Sfc_xfSZllV!}00fi*sw0x1Z7{FU73JQ~u4*+(>Pjb@B"
    "_rhI<X$90RdL1}v0Wh@Rvyxl-PJHQtbyE!loYr)gaCJikZrtRTPSi_cymp{+#<ab=+dYc{J_9*)gARH5gX^Izc|IeJQx-"
    "g?u==ajs{h+@9uHO4s9}%hbnL_+rf{msDHbfA%Lh}^qPoat{?w!e(W>+u@7vCX(x8pHTLhc@p!Mb^|>>f_p{AWSn}`@NIcunckll"
    "4^}BDf6}|abRrk@Gzr6eI?VFGP<N2$%pZ)pWH*Y?E^W8V^zy9Lgn~%nqKK<-"
    "p|MabI`PcVfzWu@<e)H)ce)XH*{`#MO{XahWg`Y!4%g>om95F=Ccl_fYe*cG`{e!>N*b1WZ(l5LhfBV}{e)pSS|LS+Y{HIUqbr88"
    "|FV4aJee~uZfBCCVfBxp<z5ew4lILIfH=q9f&wk@y2yD>ofN4^wee~uRpZwh?KmYxwzx~<o^CQV49ACjhwBMdz`pM6J{>i7m|KxY"
    "S|J|F9=kdRP{_9`=!!LgQKY#V+<NL24z4_+dw_kgMfBN+MFTeigtAER{G9tCu9szGWg~0RIfBF8q?_CSG6#O@E6lbRprRT5f=R9x"
    "hihL&pfyvr)x!-^J>1TOk{`7g?*!XU82Re;1Th6ZI^I7GSGMpvBIMUX*pMU-Ke}DSjyKldHKA7jHe)iQ@Uwr=6zdvW$d3)Z*zr6i"
    "F@A8Q~U;p~kZ$Eqce4HQ8Kc(c7M`ShC3jFBJ*WZ8n{=4`8_HKO7^3&}Mwj^o68iXIc`QmvA<L|>GJBa=nnhgv;dh_L1@4x-"
    "wH>d}c0UI#*vrN2sF66(wzG3}oXc4A8H&^WK2YflMz!G8Rnb=6N3fm7};&*@k#<!4_sD+`o^hE#aWBcm+FF*e_*YewU?_5-"
    "}7rA(EK7XHU^v`d;`sV$ga{tyldj8*^zIp$bb$b5r@9#f<_ePQS{@oX!fBWh4gna6sJ(ukHf&cdQi%<XNFP=;O`P)DJ>G|J(^H<L"
    "sem;g{s(<<X@p}IF=l7p~{!XXiSzOlT`BmS({mb)=Jul3ta_Vn>eNkuo`P%32{_Xu|&oz7V+4rBn{mFm%)j$367r*@3PyYVh-"
    "+lGXXYYRU+u#4MpKt{I<R9LC{oPkz|K#s~^N0WHp?&`H&(FVEo|EWL=l}NE_jBIk^O%40{<GW^e$^LWefzDe$qVaLL8(P%`0Vug`"
    "?sHcmiw1i<%fNx{N>l*y!-6^x9`8o{rh}V0?+N+anscIJm2#6i!b^cE{NPSY&zqQzy8QV^w+=sU%Bn=D*"
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

class Neural(Player):
    """Neural BC MLP as a standalone Axelrod strategy.  The 85-feature network and its weights are embedded; only sample mode is supported."""

    name = "Neural"
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
        _require_sample(mode, "Neural")
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
          name: str | None = None, **_: object) -> Neural:
    return Neural(seed=seed, mode=mode, name=name)


__all__ = ["Neural", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((Neural(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{Neural.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")
