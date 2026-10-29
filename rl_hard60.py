#!/usr/bin/env python3
"""RL-PPO-Hard60 as a standalone Axelrod strategy.  The 85-feature MLP weights are embedded; only sample mode is supported.
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
    "c-maO?XGppa+G;5=l3=m^<BlSpcb-dqqamFWDG6D&38PJne(-=;rQ&e*P3&DRApsGMnwMazy0U``p^IG-"
    "~Y?s{{7$n@xT4wfA>%S@IU_f@Bho+{kQ-"
    "6U;gnw{jb0KpY3md{p~;h!{7hQ|M=&B_>cedAO88j|KorCr+@p4`7i%_{OUjb$NJ~L{rkKA`rF_Cr+@j!|M~a-"
    "5`XnCfB#SakACOB{QdvZ2mS5e|A+tmxBpN7`5(T2*V<a|%s1ZLYxREr`aAvbJL_w^@2}1Cy?Orn{@pivoBHiN^_%U?%`bla+}HcN"
    "@9jD7YW~gL&)3JdcGK_P9)JDj(vOxtZhzlg{A`WgZ|k$az4vO*pSbm>tM51V+4}3X{8RnO8Sm^d?z`V<`n34@ZR<6CLc90QvnHSW"
    "cH5_)arJAutIf~8eePJVfAi^1;`>k6-"
    "AsMN9p5{5`V;TEbNBCCSC_YjuTytB$Jf@~ZMD|@9r~>??`UhDGfwkkYh$+Y^*h`9+BuJ3?7E+J^$_>@j=s^~JoS;fo~F;ZZ5%)JP"
    "Y+NR^mHrV*m}nP>ea^6ciygxneF(m`_<p>xAbxS>EroXYx`}U%y+r%8;?itK_B|vH@{EMDDH4wy(s<mTHobs_}S2<>w)Qp^?dt?S"
    "E1!F^lqo#+|<L5d(>lCW1R7gd-gpiUW2Yv&wT2!=r;9O{HuNFPIQlYzIy6$?TtHodhbJ*q)Umv*L~}kw3H@q@$LL`-"
    "B8!(@;LP1<2SdiQ@1(0-i03Qh)1XU8C^?Y@Iam)9oMv$-"
    "v2mntTSdV7%jPdpPqW}x`+6C|M#JPHJ?5xex`q?+wOO3iE8D>lG(aNJqcaT;{mc1THotUEvczb__P+{;`OrjJMUw^#Dni$x2I?N^"
    "wnB!J-"
    "%7X<kL3OLq4}1U%BpefBWg1bf;_xuY{>}qL0v1ZhBi@TZiSVXSjzpg<f91HNC{M*jl>DxnrX?t(`qPTXcS{w^(v|@K^8hT~@Ak)y"
    "~JEMKJE-;<SZ(?)1&CYh!6!Yx#Po=~e4uwst^gm2Gcz9`VtS&(#jm#_|5s>sebX>$IyKb9?Lpy(%s1qdod*Z^l}f-`)y8pZ-"
    ">1<@4&+pKq<huZ_)09dY%S?SH<j$EnBSeW~S7bkG&(6Mr#9H{DqY_v0PSj49`6Lu&&sy?re|KYA@LEp{y(J@0mNWAdi8gmg!px0O"
    "HClAg2su_$`@&SO<+*)_#LJ>|F`Eh)v<u64uPYx$PiJ0ELa|4LiW&qMd9NV>JF^e+4IN9zC1tIeaeq@birjxX1e(T6JTt)>7o>tz"
    "nzqV7lGVvdMYT0VMXdQd}`zV)0Uf^dVaX@My$>51J~$E*&mZY_Pq#F^hTxdDQtu4>11YPIW0=qw=iwg|dk(`wj`6Z|**kuP8ST)#"
    "Gm4xeI1eC8n#Ha&Z7Z>~qLRcnex+&`_X;cwA@(IVmpd$)?4iox1}gV=CpgsG;mwfAdUOGC>r4hH>!w$Z2mI9-"
    "qA{$GFnhd4&`5cHB1LbdEqhG;E!h5f1P;|TEu^}`}y6X*rSV$%xO)6?_ZTVE3qPC<U@wJF-`&|24<SMO8%Ru|8)knc-"
    "}3&B+}q7h&szHD6wXOM0{57!TIDh%oVH<x|pjjgNwp(XGXh(E$%YzTd?;=FD^D{E@i#5HQ=D=KQKY%QOWi%QQzhp<+Lu5;vT)fXu"
    "gpQm3|ke$AAZbcWY9iUyHlQy?g)3Vm8*JA1MYkE}L|Bq|tgpTl{1NmJASp})9q#^E5>rpRL0ZUu`t=JA*dqW4Lf{_A*R$m_P`mZ{"
    "LS?qUftHf0*U=c+6)DqVP&-i+MH-D^+_j@<aNV*oTV&;tlQEx$K_MX~{gw0rvI@7fWwL=sbIey~h5ES}johp*;Sj}2CiY-"
    "?sr}pEm_t^FNSOMoLkn0qSwWbBXI3%<(_0m3XCZ$Sxu1hhG<y*(x)GF1Z)QcKg!MSf0?3Gk-"
    "u<|@~Jm~MW?AiHR28!$2X~&<bji;BWuhlE4xIDN(?Jup8!(MS5VskGxy`xw>OTB$voFc6rieh#qTYBhQ`*QpIn=`Y9rG2U6NFkYg"
    "!jJPRIU$BCUU0LSEGe-ev}iFY7)Kz`3gzHYSk}R(bGb4EeUhG<9*dsxt8XIEv50!p5q)CG>Eu%6X34TeeM5?}%75<Ck2nG{`P)Pg"
    "ow2j8M@rzoKIBn4yY<#M6(Vz0TBUqRxr3f^2BD?xtT6N`je4Ysev?|+Pn%lrK$nv_l!Az^`{|o>YHLyDQcw&e1>`L%A=-"
    "JKYS}1F*2$|JNzpultx}$$^Gs_@uX(*leHHB{_pW`UW5mBl6w=CSm!RwS_>|H(&(Q|c*8eN^56R5oEK@X_aY!gY=vgTQDcRIZk9D"
    "QfUbm|^s}oTfga4d7<s+?7EaAzWNGRC>T1MLAguESJq!`NS#S&y=#D^-"
    "lwN3C<4xtws_kENO>EiX62T4rk2AzkjD6l5c%ezv$L+?tvT6e1u>7}J@r$>43;TgRQdQYo_j8v(_Q!I<MPP(mY(q`93^veD86nNA"
    "6T*6Fb!+Oqa0CtrQsW_c=!TL_!EU`@M(Ff0DmFk~-"
    "M*6;$aVeiuAX6CDvRiSuabsGU`h2bKFLGfv%2q|gJJUnUl1O#aRLGltD~ezl$@DF6aVTkJXE4!bRvJL~)ma`ps!?Dlu5Z0|1?M;w"
    "v_Yn>^YGo=Pi%1#cK!cJX7h2t5h1i{2dB)TR?0+0sg2Ge{+)JnC0WX!`=IQ3t{5WbDgJ(>D2l-"
    "Cw+40Qxd!>jm9kt=98(fa{>7Q8f0aihbqj@oo}spC+&OvV^0BV;Gh#UFP3IQHj~3m?eWp;M3QfUCuWx4Jr~S}4%Tzl~op^DgX{~E"
    "<>emk|tQICGW>>zTjj~&;ER{W+ddeG#5)m790BG?kley$idqi@ol2318QR%E$R(dr`qB>``&h||Dl(Ol()?y|bA9awkY8x4k*5dF"
    "}Q`}pmQ92D(9)1x6l@{rUR3*HKz!5BUU}`6D=4zF)wY>rf`y}u>0~AN&7j*<t0BH4S1HQ~QCwb&lX11!H(1TW@uQQ`B7l%=FbuB*"
    "~IbAi0j=2n<iyBw&T>FU(EfWL15dA%MkCwpqD~}ZTb>X^m?S;scwZwPi%9C;|;?Sc^9;$kHj+q6pzqFoMTeG&eE=A!&hu~o|x7c`"
    "kq~4JFS1nhcR<{xWT{0!on3a#~y^`~5ubjFYE|PRXyG~o|Yx!cgUNxuXN7!q5-"
    "<=Q8&gC5ZYB%aNle<#Zlaj5>7xV)4D7q4`lh}x!5yv8rXDM5cvQY0#|D$`LZ1gz>DsjjbbbhFYjEB((+D<sRo}Fiwo~>RRM~dEZ-"
    ";qXA;p!8VghpaX@aQ~Ng%&-Pn~{NHR6o%{GV-KXYI8{ml?7^<&U%Fx=wlI@6oO|~{m$hS>2$WA$_@2S4jHu`P^Zk#;zGwubJnMuR"
    "E)}}rj&b=h?10?HM95~i}D9hh3H?8T~!Hz%!lyKx>Pr)<HX;oqR|mppG^Q$LZ-VU8fF+{O>8!~vY1#2N>V6u*?=S8_;Hb~gkqDt#"
    "9LG>Qu#G}ViPbkdH7rie;`LYiY@H~>MpG+p5Ey`BJ}n;r@GEx?eci;iVUh(FUhk4Oy8B>=}f+l)JZ4($+*@jZ57P5UG<+CK(+AoS"
    "GuQ3;T<LNAs5nuKZgstEyJh6jRKQayUOd`a~0{dBoQWu*9MA=Q;$erqFh9~G=U=Bj=tcjN~Ow<07xpNokOOfjebVLDs1^@oJJQ~h"
    "gMgl*A~~>9C4@<jvsf=3D@HTFGY(I)_Yy+#(EjlsH!jYh*g>A#+}O*lM=KCCe(K&4~Gn$9oXmZRM)GY)J%p=?u-"
    "gfVPD}?Sqe#3;4gXxU}0J+19YkRq8h<ZYkQHV>tzNMr`@jSrtC={svAC;ROql=&SqG#85<QDwdu9!b!Xc8aizK{tt5pqJ=oDJ4OM"
    "_r4W(3w9eSfg)<fX|>xk0T>6%8S1`7O#JcojMoyb*m><wqqE&(DI>a{zM*c51ant?31YSD3|g5l*-a9X^*I&^3y#7Al8B2S-3R-"
    "1SUXiEdfVRiaBedGsve!7HB^_W;<V?%Rdd=%;Ni&Uw)X)QV(8ZT<g1-"
    "kVB^>h(0f>|TD6Cd@qXPgzvlqdKCXon)v{>8$rzayn~rt3c^TT}Y5o3ZBD82Vz0B-W22_KmH|u|poJUo#$-"
    "sES_j0&!L9*ef$|c&7AFF&gXm4Qi+h5_d^OVgofkQG0@Cl*lHBt1@+cgl<X?$$zce(Q&5Os5IuS1hXg{s5E&)Rgf!Ge$M}uKT|qp"
    ">7mfSNUjNfvDZi|6d(y6mrNuvJ1rsrw@YMMiP?dvd#9_$q87+XiVA;G{%9q|-"
    ")nztP&frB!qU!Sm84aRgOvGxc`A{TlkO|)(jtnQ0<dk7q><U|{6xT5?I_(gDNbDaq2x9yX+R-sGt~@K1d6-"
    "enUc>CMNl1dlg@%p(S(Tf7}!ILug?Y39h47{(FI8#bucMrC_XA~=^^^DQ>1tDMe4&Z4wc6K(&6=h$f)iU&sBb<H>@BFM!jlF12<^"
    "R>TJ_#-s9_4D{Bi7U-"
    "dQGwDIIfkM9Pybl&cLN|B)V%_dl@B~q(Zty*4>T6dcVC0Ot_d`c}Bc|G7A0h4o8X+r^nUi$iCCFn0}P+f&G5xvAkGV~?HrH7+!s*"
    "S%Wm<<t7yDITxktlV{IIf4FGvpXd<~|e37_6>kAVmk{&Tm((rnjhFrtCbB6umdS1$~XKm1p#2IMNaT`OtFYWk*%36JDF*_-"
    "u$X$l$$hQh`+UXm&~fbjheT*>DH*k43DI%jVbBtJKL$QVPvhVfT_V=68cFa5tcsBVkD;K>D6eK14<3P?m?e_OVXk6?g!B68(b~Ck"
    "a*s$H|lB73&^HEKR;on^i~EprMoq7ML|R4Un%SqbKM=BJ4Vg*PhQ<Yn7v3+ef+UiC45((fY+nPbim>>O_4-"
    "qY^aG?P~6GOg$tZIy1lA=*oZ$ZVYbYMr-"
    "a+lIXBfRN%1A<`yeB;BkIEH+m<t7}Qe=6qyQXap~!5!Lp(fo@_)|7KwhJm!B0Lxru^B?^MAF1}|1MM<ZZ-"
    "fh!zG#2US-h$n&Z>zo-RZSl7q__-5vbf}VN#~-"
    "_fxDlUeqa+NYqFrxYfqBwN%O^~+<mU#sS;?xK4Ubb$8!KUAdKWso^?e+lv?3B`(6?W@1~WoM^w$PklMd+C@z@JQqHP9ldoxp3O~R"
    "V;ddMVKyJKzVSo+xu)S7%{vpgxxZoWry6MC+0#`P=*X_l!-"
    "Q!4a0ATl~Db`Z6d@#yxyp0Awlln&_5h6!wk8of=GQ5P&VaVFOIqGY6j@nwdub4+_gXTUYq;nPnmhur*u{p+y?gafMG08;hFGTmdl"
    "F4hkp^ITw3+yR+r|0&&DU#38fI$L+>^5z8;rxn5}Mg9I&F{qQglV>VpA#I7^cqjvmacP<MY}x2OKJs%dSzYGW>WAcbTDc_S0T{AG"
    "8cksxn<sYtbvL;YM^x%Wr;XzsG?+83Q{Pm9*eM$o@Qzy*iH>N?Qv*1a&Q>ffU!Ip%_4FH$LUxnTp5?a@H3;?7)qiL;0_JZ&U_GqP"
    "o_jP@iCIxMUWzI3P%YTXBY3O8ZJ=i}f$K;uFD<r4y&l-_rD->TGDwVfmN39$de;VDvmvKNMc<DF-"
    "ZGyfWuTw9^)yDSa7Q<10`Dr2XQ1!K0XbN-"
    "gsa=E?V|Uq@3~}EQNI&*3DAm4mne~~k4_)=xTim}flX#N$iB2SM%KLaTXk`rz7B^^0uMkyM90awpYK-hMqkQ4`4X4W8R14GyB=UQ"
    "vwd9|@nf(@vrWMXp$(@GP%6`-"
    "CV!N+lV<87(pO*f0Ig`hDlN*6a$_Z41IG(w>t;bkJB#)nz1(b0y0gYPp(;R+DoZad2yUmtF#xxZno1Szql0y5vPCtg$FIbOV@yeZ"
    "zdiG1iabR|T~OA$R1Z32nuE>yDw`xiQ!KI<Ne8kf&`bQtUceVC-Jk`;c6q?hJJABABU-"
    "k!w?44i{OMsgQ~K~kZX{*;UhNb*dJ1UyeyOg=mZ;u!0PO%Q34IE!`b8pI&y!~WX6wB#aI5+nefMmDA&SJWI30_=(($WOPLO95b-"
    "hkMP|Po4mVwdezClBI6@uxL(dE!fY&$c;MeRbf6wIf80VH;(qx95t&_y~3hNW%M>EA2b-"
    "+Xe;`m_psman=Y<1F+W^prEI8`A^>)o}+FS*_;hcAa*}tPDsC6u5T-"
    "36c~>gwu+nswYo}KH<wG$)fZHTI)SBLTzVVK4sbDy;L`5nelTjXjn$qM$1u=lnQQVXMtpr4WJrE0W7oQPU=O_-"
    "n}YLo}iYCTiUNxvyX;}9@`^Da60M>pYS9WNDf81SFs*yGkO8vBvFp#vOv3CPe@Dl<OvB_1(FKbFyoxkm35OK=;-"
    "c*+Mu#5`sd(+*~NN(TiRAyN$@}_uP46)?igHaVmUN36)S*+RU+~|ot}pxBhBHF-"
    ">r44jF~K8xVgNttDw~}0Oo<G;w;t6)>hMY*>3-"
    "}`)l3s4hSA}0rRj>V(6n12taGzfTGb;upOm;tI&(ZF+K<qhXXu%b&mszKz=#<0|}>vt`OU3^u*Z(6M|EUx1=3VC_HfVVR0#Y+KGD"
    ")e!AG}Y=FQmAlA4=!;;Tqca!pv>*`3^d6;S&fx8UW^vN2waddq23iKp7wBrQeDmI_-"
    "`8bFw#|`5}_i>)gX@I;HKY<QrpwW;|CLP$({CHyD!3om2r<{`paU6w*bG1`6(6))Y>(FP8Q!@5h9#Zs^d8XRpe9m8a4(p5IGm!><"
    "St)~Tn$@NZZB-vqi@}Xh2u*qPkh|Q^aGIWcHuqsE4Cq8U6f)!Mv}S;!l?rlHWC+v&s7OCZbafjoJG}(Y!7kQ;9H=Pe<f5B%D75oH"
    "QV4=_Vg`@`W~Qr$4eI}0JgiKKl{!)N)Ao6ZplaV}3vT+^Eq_l>k?h8-"
    "sYQC7r!%>@k|d&5CRzG0=yXbSr*q!GlSn3bKmfBrw9~0V3z2xJ{gu@#AiVCH=EtFQif1tZM;3j<#@bFgU*~8$<cK%h<IfB|diDwd"
    "w8VnCqcBM`k6JNvk`4r>MW+oh64KFy9Rj+x0w=i7TOCcyOb23C<@anMtt;h{k0%#L+sD!NNPoz7R<XiV1uHLjV6wSVo{ombss_tw"
    "MW7_oYd!SpO&c%-BI-"
    "1Z#>hiR1qc;@9ygT4T=x)2C?pnL|7=x0S_hAUzi~YHXz0P}QY72SuIR3rfS@~lxNPhhV4>Hb$J=@}@2Tw+;<VXKM7z<kq4Nb&r(;"
    "T6{zZ9AmGSs4)0?sf`LX8p+mlGFy%DDaye+fT_*c>2R)8KlblB?)qDFA7+>C0_K{=5k8$S^KNDk>Fjrc`QG#}kiDGO<Uq%hdHDyj"
    "Hs038an#sx0D5Zy$sH5zRywzWPE^=mv5RuffPV-Mwmfl&HrSPdvkdHnKb8r_Tzf2YWJ9M6*@x&LtF;rx$wsa^mZCc5yzlLwShcbT"
    "b2H?4D-iNrbhS3q-|bM(boSrG}eV}Knd-"
    "~+qa50n1S*3T(DKesq^932yv6wI}oeAc;BLJ9`(@#V#EO2dCNoA(C7GF{DZI1KvK6{{L+Wx?jvmdw5_$Qa#xxJ*0gib*e90R?3l&"
    "6#Ru#xa-iWb1+b;hhUdA`aT=<_wJ+9hCXqy5j8|kZxN0rCW6IgZ!ce6@ftY85~}=kuss&XJokBsWQlX0vbwoKzjjhl<z0Paygx}p"
    "6T_*0jOB4Mg7?5b5+-"
    "v;H5nZ)mqC+%Pvt$umW=0`uiv`XtfxdYrX0vRkqtrlfgGeK0&ix&t|Bo0+on8@tN+bM1y)Dt`gz`^iuxs?KivXolbN!q%ks^zI5O"
    "QHMpB8SzUZ?A3zdY*{_JQQ59D+p=M+zPRjC#Omsq9@&PENU4c;!<@*Nay>PZ+4&*6#*f*S6aHU%QCTR(<A5AEpasYqualy2ov&9P"
    "Xi={B#)XQ?8fKQtpvTwACk_k#8cIoX<J=Q@+=NHKJ89MqaXsFM?x0B(>qCU8B6g$Vg$Q$GkH_vG>Jb<KZ!g5N&yP;@Pb(I8*H(c)"
    "zfVj0IAz7rtR|nZ7`yT~_a!<5IAPi>Tu90ie+uhYY-"
    "7q*f$e;;)8Al!@0g@yl$_?rmxj8^2*+Z{pba$VGSm~qAI|{bYC7IQ$NazlDt#g{W#XO>NGxJwp0$s^%!l(?kf_j4*4UHsY5I8IO!"
    "#3sd6vEbkUm?l|Y)+qW10K*~%!deW;Pk6Ovxz->fs@?4;ol@>XOcoGH;0|Qz69}*5m5gxfZ<s9EIV+WM&-"
    "CqwV>$@H85d%V^OW@q84Br{L|K}O>*gbC<$jdW~YuY<9Ko0)xRg2K|sEG`DUMFo0KX-OX|UV|Fvz{$IC|;wbY9PYjUV)GXVqTAp~"
    "5~X|~Zn%O=*OO4K0@ru7wBS)rdcD`-hTabEWDmXyy@J!lO_{y(2#*3F{M4h_8s(1}H`-"
    "hZg$>{sY55b~nU*Ffb>!E7KYkx%GA)brsxE{*hPmFf9#_EAz~qDId9s1%i~Ys33*l}xiCwHV3GgV#&H8zhK`8p#)s9Pv>fuumWZ&"
    "D2Y<GT^Xe12vlSaMmailpO^vVSnCa#yRz9GlPPV+vtmo4sAC(CVVAFF?IYY?oVtU;Kl@n9pLnju!m0KwT%Z$cbHO7k3Cii$-"
    "F+1UI{sTt~@HD4p*C1Q34blJJVDEvZnxcVcBRoG=N-"
    "G68@nAk)o<j$UsXnAOQI242dtw{xp5X3EdrLWYkdFwwIWE+3}fSDqb~zfQb&-"
    "9H3y)_G3xuf%d$}&oni9{9qF=11S^&nz7F3ZuQbT7#fHKR75@EMCa(4gXhx9nXx_RfyAwqhC(DxG>|c<bSKiqJUI0%wC8|XdNQXM"
    "s33(rM=*5r5K?ISA8IZg11Ebe3Ss~shVD7wW6Nn%HrP;F*QbhJng_gu3A)s?3Nxw<KNWCpXGl6A9snD35#xB?pI;2S>&LSAATm&|"
    "&?tGcNK;geAEyk>l1L;;G7S6t=>{u2n5fm!!mL)WtW(1AR5@}|3go|{ht5G2e4=Q}LBj<1tZc(HkiA&36Z*hU*>uQjwfFjs1kvq4"
    "8WG!IeSpO@D#@T&IOj`vChH9H{gyP8#p$L+2{-jdG=wg9IYNQj%rOpf>Eufq5+Dd(J-"
    "Jnz&LPavH&3Zi;IWbn&?L)D6lC>Uu){e&Qj>**s<|BM=FkRqH!OHD&^&jl19TAR3iUD+J7EJPkwqCA%y5J4JKh<xIFw#m=jgq>u("
    "an1+-Rc6_D2<}kaw1QOS*{`M0@AfuqR<gWcCj!6&5?$PUH)XbOCA$Y{7^GuY<M@GdU=sledp#fE<&iSusEWm<Qnd`<-"
    "+&3E(c!&{U+1{otMpEHJX9QMDPMtZt56w7IjI^#nvD5=wpk{i;(4(+oKa6z@^E2GYRkL`$(SG;^w89@k0JwdK)d<{0dbgZ?YJM3K"
    "(8;_bd9Ll75|1E*TM#jVoSJRtCK+|3GVED(vVnhx|akIH@Vr9*<$Qv0-Lbf8!53~>%uXz|CK=){-d@-"
    "m+roPoa0Jchv(MX4f5=dHP+lMF7cf_G?BL-NRLJft=h;7aI5Y_bh@pm(i9jt*5UbWX5M%5VW&@|9`vKt5(q!Jr)_yF{FT5QSz_b@"
    "et#g3eX$Cb<x7@dl-"
    "N>vnf==^7LS3ZzBQsRlTLx(9eOIbZk5!wh~fN5^6z#DyXwL0_ZI_IB`y5ZggUg~J9jxgrUNudW2t{Nw`eK&uK5jg?vC6<#SnP}Z}"
    "v2|IZ|8+=^vtR!266%MBXKcpvZbWotM-svIM!qD>8_YO+Kk$uKP6+)q6iD~3eix98f{HaV(N*XCEqQz|<lhWHS4mWzebm$K$DOq_"
    "QMf%py41G?bASZ0gN*`p6k?Os70iz+wfnP+k_Y%b+8Ph3EagBRlfbj54z?G8VruCyY<33L&Tkzww`llsptMDcI{fL+@SZJRARFVU"
    "{lkMBG#nZsM7RnN$%}$7t_%atd%8&FVJLigY8>x~)@okmSd_g%FmJTbw%u?!+!s)>`QlzW{`+?R6K42zaoS8@%I!sYwYv6JdwyRz"
    "S94gMv)SGseIvqCei~!+iAk)qPyxG|{y5NX<Z&pmq4D?#}bXgUbQ8xUlCK7N`2Cc#%Q=_Hj=y0|ZAqQ?I2e3W>!C7dVA!I<(ywC="
    "AuT4Vy0a$oIPr(f0p~7>4sF=M1P6rxL7g*-yQr#rSY|#!^CC~MvsBR>kVyxH~WcKJh7XQ<-"
    "RD>JNb_Vh%8we&0M7@V1f(mg(#rC{m8wW2~QsfF`r6dc`Yg_W|V9Dxo=s-=VGkKZx0H8~;)U#a-"
    "2nxlN>LLy!H^LQa6%mP`8RC&OD=sR!Kr{2B&_A<4I0Dg~1y2H&oJEwp&CGE0&wE>c`HX8t<c7gI0HE8d_(wW>sP~8)C8)D<4VeRt"
    "-&34ga!=?o%3TU$fvl#FQ;2{^&~_q%&8%*eRDJv?y$ZCD1psg;{WA8`1fa3{uPVj?6mFKIuI^@!kfH!FD-"
    "q#lZ34vFXp#@2d@&o)Oo4$BIdl+5P}L&QeR`iF<O$SP$0aG`AjN8NK^ts{ild{452Si41DbwsuBt~C7xXr#b;N)j@oH(^Z2&unAA"
    "x>66y2oMkb6Vm37oP4t2(;klgAUR5o$=@L>zd{9cNShxy_!y2@3EDEr%{bwkhCq4=NL*TM^feDY}ot3bsO&=!firt}M;=NRJMrZe"
    "*$M3<sF5y9ZA^$f1Ekvn$m=nyun%zf%av3As9FNi+<jgliT!=_!8Mo<qn5Fb{F)Rh>g6<+?(vqnGS%=^R4-pp~@%5CUbV8N-"
    "@_?)i<Z`l0T@%6g<?GnHeo&yauONbpKQx!}XnmC$#6r4qH;aeyJEOcwzHwT|`&EY3h)){b`QfJ}T+{sIFMArPJUxhm)J^CfU!=z*"
    "~;hS_Hnu3j@$-"
    ";t|RN$BB)hJ`X}gKk^XGj}di?axC40J3;#@Aa9`J76o2DW$Z#0f=yF0@Y`A=n8`NJ*~1?Z|%|_Rl?!|J)l2CyM(`IW=BQOWg^_Mn"
    "seosLnDS_df}O^%<hrv5wa*4Xdp(rVPO66AoS@&s|oNK%7ZMy=9ra~0>)YVqjk7Triq*p<|2uIk=pJwbwPzQh(ozHEC>PQu~fgop"
    "mxRus-`Hik^nl4EJARfct>QluhRle5T)^|-"
    "VuVq&Z>sIxqf<aJO;Q&%Ir~v2<<xLiI_`_62d^o)I)pS^t2L(BNS@xrlN!_00Wf5RGdKz8+<3#ZtRU`$noYI#V&$`2&q%H8j-"
    "o93nhp6%J4$_)~YYypq_UR`c3jdsQ{ItB;Kf{FO+BV&2w*l$oI5S5!-"
    "|Lr^^W~E;_liSu<o)JYI7JuPckJun@y2&B5y+{Y)0DSd(&h1uq&$AN@y_NRVOTLdhj33N4oHOPV`8he5r!EEvhvpmL>~8OKi}$SK"
    "CS<BV`CDLarAu4!AGBl&oND*#3gz0Hwf4p0L+Kflect>*i3sRjA}%Y@@2AVSsUcBqFks5CoCz+L&OIRpGo2ExAE*?F7vyC*50j*V"
    "BmDgr7G7sAiX*lD~G<dIB24+0=(1UnR3cV-iOSho%XoO~kjtwZOB_Vwd|WWKFm<O3nR-lEA~=QS2k!@CYfqJ3E&y#x3`FO^>#a0H"
    "77Bdo1tg6MT0x#VK?CzNgUp5g;^r&jw015sqrwfEA)j*#Xn<>VlW)q&|pbo=L5xixeqEEFdCx-Zp17xD>-"
    "P54bz3+hIq_8B9{C~fi7GWDN)IN|~fGkEe8R_VWSNz>sSd=T!E&`#1o|4w@t>3zW2W~6U8G72KuU@?8wO<pc_#!fYwZb=!ezv>l("
    "ec~np<rzRHL6`^O*joULi#T-@FsPe0E%n${^xLeZp{4r8<{OakzUCN4V<F)_q*T0%W*ou?C=@o{F7;B$hydbU(1M3X>d9G1?-"
    "Q>0Eh7|qO^~FBVtP*}<qi+q3UV?SH#5m~+6#}4f~$+3I>80p$HzkR2_^(iJbJsstlH#&d^TL60-"
    "3w?_WC304$I<KYS#G%RLTk{@bKEsJ}sU@U4<9QxX3#%plkQ?xGy1@P74pP6-9YMLbl+bvF!$RvhtK_3|cc_M?|lH_QCxEgzm2-"
    "p!^2ht7(Ij7vju&+^KaFy(KKdF==3B0NwmRG&ViY1hD&*M2xl)7lfybD?7b8n?Y!w!#FG52hBQgFHW(fKHzl#w~hY7Hr5>}f>~oa"
    "%jqnXh7=7QJGMcp41g8G2(U|-7Co1)DWx!1kdi?B&B!>P33-qqe^45x0FzEMLcSdz%5!tO4jEeduW-"
    "$vAnkgs2FqmtL2(UZ{RHOJgK+Kmys!Kwz#Cu<>jNLkob0{fweb0TLxB*P>gv#y(Ptnd{Yw4{h1g1~NSd9x9i#(rTULr~sD24IMCF"
    "m;mv_egm;@Fm9r@C((|7VO>;qye!(BFwj&;40>5VFMZ1Ziull@cl%Me)DMBhs<8r+)*F!oajcq}s4+g*#V;!`agXZZP!z)$Z5RR)"
    "Szblm|P87Aiv6%zw4Ks%+72qmhWB30to)Dcl=?V)7O$_@t-"
    "P9jkCN>KIJk4Wn#>bDa4(GG)X(%h5F6MWH0!?<SRt5Z7^vZLO$F3nY6Q1uBzitUtUGdr6EIZFQ14SxqY2n{Uf?AKE@L=(_PQmQv7"
    "*%E<(Q>5JfK|f74)vy;90q7;NMOPbe1=tT(UIbV=(8DBf==Q;C>;$>PW_q<ZJK)@+;x_xKT0Z#kEKZBDlZ_3IPHkspN78j{tEPYE"
    "&AbF)yA7PUB<+LhAwBO?jyksiW{5eb3Ba`H_ld(?xjjU>=_7fUJ<!B@=%Xz{ww#@K)=o!fzb&PV%hp#?JMm_pp^MIip++Im+(n0k"
    "5kZo~u?ZJFEd&se65I*r#RFH`?qXo0${GQ#b10(><mJ-O&$UMyYcVWaD{XbamRYf-eX<=%TjuMbeOj$(vGiEzCrmu#lJa-MqRl$c"
    "QI=f_mf-"
    "r(EC9E9DI<{hYt2*MMS~wYFN%E5wB#>r3KH$xZs)33mFJ<mM_Qy0Eg1sFSa=Ow(>0=1OL+={X{)GjC)i|D4R#!as$?NNNYyY0Ovp"
    "?q9Cb0RJM=O0*K%A?4tZ-vYXa9lVcfu`i7qjDDgvZX@13L(2BQZ07Uas{dky^JlnNsOH5?vsbP(52B5%WpL9%7Buz={BWO4v~`9K"
    "=kH1$W)rB%srjnLOizk@)Mev;Y5!D0O%*FrVHpGVIreW=Ljjdi@rf=QA%j7)UGhc1bUB%Ls35OgH0#E#}RqkWR(*&ZHl+oY&n0#X"
    "QH=w+zq(pE{&y0K})S}42QI8vB)W~8*I$swny6^ih70&0M8qQz&4Uoc6K6E8#!6mt#R$A~q6DEN_3-"
    "OVc6@QUM3Xa!N7WN{AY>K3r=L{cWNEAQH%&vuT#HWI;$>KgQn3^kG{ngHzRYt3o<wN#Lo9&376@hjY3A>@&eoP@ZsnZkw7f||f?C"
    "ah7+&-DLWy4p}6z*d6;@Nh`V&m2%n@x_E^KO(RKd_twsLecvy&}*QFgfjr}Q6da?ge?u7B_-"
    "2}p+W`O8x~?j@nvU^#rA@vyVD_zlN%2S<Wb3JeKzl;al2pA*wA|LVK~NFCmdXC8YuL!wDMopygl?w{2Wa|=0{eip-"
    "@EFJ7Oh6Hl$yK=L0P0mNhRd;E$}03OV`f+k_@%IkXTHulSzTJ2;{c!ji*;Aw;u(QN=P$i;QzvLC}jD+|71(^_KTY<AUeF5k7uV7?"
    "5gBH?oq!=_UgCKWC&qu{^fuo?}J`tGOa1?*d8%OuL&h&}Z`AbcE(rry|}2G3p$=A(G(*wzTQ7fDFfegROvn4~Sb&y?3)I1+>7$dy"
    "ZB!`+-AgVyK!fK`1-"
    "T<bq2(N{Oke2!6KGfQ>>ZU>8ym*TocYRfM4Dw*Sh%qLR^<Xwaf{Q2c@BE+Xn_f!Qc|v+EQx!cAcobvbA<DHk9OeUai%c=6$TXnmX"
    "yIxYjhE{djn=x!zhMB5;;Wb)=Hw&>B80~(D_e1w3$MzQ61ko3LaM&7DT2lOP&!5*!!g6l|xETALMtz1}O#^j}B{zcGr;~a?q2wfa"
    "Dl&7SD*C|0IJGR@P*8x8v@LdXRdK1w28--ZuhH-"
    "1zP>ad7%8i38{sV3R@%)M!ihHL0Wf!D^zp_0%fo`BSz!Zq25n$+<9M=A5!ahn|{ty5oYnvGocQW@${*kceK#Q?V_o(v)w!yhkoN)"
    "hnqJsefzI?XONIE8Ez8xgkz)1*L&xWXI4|x8`OalI1ChY-o73to20xgoAcnZ{<gLBb==aT0TA>W-PPK?wH;R}F%i2`h-"
    "s$g<b(kQ1uzN4gkWUe&AJ{C_lX<bo32P5hEDdLwMvB-"
    "#e8Kj~M=@#z*z7$)LCgQM%RyHbeA+(Df>D_#J+Ud1&IFhG=D(*__flZH#{F4X6B8=m8X~Fu*%8E9|wprq`TUqTu4sH;@z4J4nO*L"
    "${uzM)SDS9T@aqMh<#*r9V?lfdJ^VW$?upKDv*PlKs*|dAfm)?FkPoju?IDw!qj~z&{wvK5wxm=xuFSvfR&&tjQoOT6$!BuwhFbK"
    "oL{vgS=yWx}4jQ-"
    "5jFdWE}aEal}CM(Zo$b=_!*R9kyL^WD#YZveb{m>0!z0g}_#SHBVf&8lmCccB<8tyHZ86BNsgwN*S<{AJkgsudnek}lu7A7uYHZ5"
    "I0eUWk}#JA1oc0wY$i92nLiCI?^&Gh53ETgyG==EoG#5~&$9cC1SYy=%UYC9ri5w0H|AXs6jilpp`Z1@5F7^vx4eohqgHc5KGjG2"
    "BOcf6ZR;E~EN&UT<Lf|GXe%UqMHjjk8~ReD6A_D7#MVL6{y@`XjGu^~hQVJ~}*+1)s%(yqS6<|$we^>Jr;Lt2Y)GhBRQ37Bpr=HV"
    "Q0sJBzqpbcxj&jq^w(f2GthpdGqLC|#xbvE|0h1t)62nbGpOp75^mj?$En@^CA{-UT?moXJI6-"
    "gkc+haGFg@QTIF<WMkJW<oG019b1lYs$U-"
    "gGPBz;$9`9#ynPAe=2DeGk6jOqw%QA}7*k+C}z&2=*p3Vp)*!mwqc@08ggWnqxxh?r*oTSDE|q{%wbWAw~ktK<xC0v7x=|=VH~!B"
    ")}ASBT3?f#&IF`Vu}&czGp`{C>y-"
    "S6w8nhQAXpi*9vPv?}lP)T$3CU52yc6O+<E<!Wv`+5O2S+QXy%O8(C5Vjoy(K!O16uAXma6Ji2+Pmd`L8d&)O$qWeJ7qlJlMWwLA"
    "ubjv!6S%y<kb_wYmb6>azeRe#X7IE4AJv5&S-"
    "8vYxj;UY9LA1cOb+~q|ULs_G;pE1wh@D9TP%Zuo!=0ciEIJ1~1V~8lnP1?Eb%`ZtJHE<ppjqRjPVTX?KaU+sv+W$t$JL3jh?FQvU"
    "+x6jO$^7F?oxX1nV9iLs}0m}p|<7k!E1*90t-IPy0qwF$+~Gv&`&I9BD91$3Hc-"
    "7K4@>)Xt{dZ8O&&BlTuMM#BmB)ft6tIfmPPS@E;!?Qc{cT+vDa7X@c(LPT<v`^raxQu4QD}j7@Pfl@FC#@o#kM6SpBpl%^k=F72a"
    "_4kf@t#_Z=rm(z(JR8&f|%uFbw1dTEct_U<LC`&}z*cqJ+fw~hH-UI@?_#b_H8Vfz3dTavpK<Q7gQlrLe(P}b8o2MGoT9yzZL6kq"
    "i7Gnc~i&0*tX12MnhfOV;JzHcCdoD<(&<B#?$w3E}j527zlb`8bi7OHK(z4z-"
    "p9bw3a7oNFvonJehSRt4=lGmxZXj_i0Idutbck@W2j_b;l^ARXp3sCtbD~QCXAa>QGQ(`z1C_9Mh;la_GQ+295m#H88{m&eNAS9z"
    "^eVY#tKl*22^H0bb8SX<<k!i9_J)lBmY82Pk|$|_ZL&z20|bRNbHdmhz0aCI);l$0<LoV+)U<!V3D4zTw()jOJjlQ&3gu)tqbjj^"
    "^}S`14cFwE_yli1a43w0l~&y{aEp}T*o1~0fp%9)CPAC+dXHIHu_8L!38pX5z??rFh^-VMcC|{^nUX<-"
    "WgXlvNp}cWP;U;q9H(WPfAydedl~HQA`r^w&+^6UNOczTa@T!VwTy&-M*GA4tIDx$p&sr(VGm$7ZudkF#rpLl#&-"
    "M2xcSvm=}PR<0;Y6?(tL|Nf!Q%t7Q)&BBOob!@aY4(tWG7xF4i=!()EiC&cU!A)MunL4tXP4pA(j-"
    "#t>5I;^B+Ac^V7Gs5^lFD*sa8SRgP)A#CT%u_HR{kWTCM=8?b{6@*b8B>uJRk3d)@Z8YyNl?Aw(JZsZ#&KQ8ugfkKoJzy&Z_T1e2"
    "p<GzD8?~%v+DHP6jnV2fdN%gHw9C&*aX7gO95E&JkH%1f;k187!zJ(RvJl191{o4dj+{82&$E0Es|6f{B3#iCq%B416yb`0J{AxI"
    "#s3m>Ncj+t;F+ZLB<0PrTa>#tRp~lKp|`@2kewb84<yR-aK5Zm`K=4SiIzbho{E7MGX|U=&|3Pbd(AHcBK&?l0Sv*-"
    "BwS_;C$WXZ)Q&qu*Km%<I2PocB!${11DPd6&_O^QGlCwNu8$_@E?aZjKixi2Zmk9!p{x>HxI}ng{0y{%kv#1F)Fpt(;Ee_N_GMcG"
    "q~Aso;hhq{7&24s9CwJyux9b(Xj*?kU&B}n2;nnxJP(Lx!Hb>J-"
    "B1l_uZIT>eB@Aq79a(G`?yzDHft!Yo6VbD(I*81zC2JMb`&WgfFk33D7Y^nt4FHd?O?xu{uE%WrefnD7`LZ)0RiJNHU_iK1W>H$("
    "=KIC_Vts2V<GPu3+qR>n&GST92;6Pc|fN~$Dqg-MFxr-Tf~lis~5LHwJ2%`d@FAn-ho>wmR@p{ne_kyNIQhOw-"
    "M(PMIkJHo6xmXGQl3F1&`_kl-"
    "j>ZE=)Cqn{hjdlswE#5ysOVD=H0bB@)AdpKtD`Yk>o_5UTs%a2?bo&GuE$<!04uk>TJ<aPkp1z!tmWPzI-86Y>-"
    "z@L&e{#||nV={UZAfkR_JN#g}4roQiwK#LRm<HRH`L{dN#@j6YJyAq0ZUF(tW-AtsE3ZYWrps>I4Er)|hn=ow{v@>JVaxp+Fo;y^"
    "YN!EyfdLXbVAP=_PfYOQW$Be=d(I8g&fV>>-JNr{6HCr8@{(=5{T$#8oLT9K%-j8?ur7x!;VpI?|PH3Uz1t7y_>a!N?70-"
    "p3h_4ki{p?!trR)KqgM+<9X{5lo*z0@VlGgBwup;29aw%|pAGbu^bBMx{s221>l+eZ4=-8`LuZ|R(;j_>nt|6=I1$e-"
    "tpoTF#6Xz`k0l2&k#^F@Ae#2>^s&w(ODx0DlM}taWSZ)5imC<y*plpN$ZH7w~>dTwfp>#nI3}{OM))$0A{}l5?8Utxciy#&<eP<="
    "*I2cac>u$T=a;9Bi-@`m%Ue8a!GtH<x#<))&j{wm~S=KS?7$^dcBtSyI+2vg2fhEV;$}!fJ2qJj!IUylk=~y$YWS%fwd-"
    "4J)I&fm0HeaoDm+<HzP!t;XOA|`!pA|a#Ny|Dt%gC`zo`w4H!pu&eiA@l)R95U?=`2f{0sGP8sG_uEe%Kn|GL&-3OegA^-"
    "dlFbSfD_yh-Z-0|L(p)SY>0KREkhGxcP$F$AvF|Nt#VpSDno8S5S)sbe7+ZVJL6{YG}al=<!b!*$}DhEL8THPFm5Y-"
    "d!aBMNSJe$pKR1he!h2A6vA{cQGhvwr|6pEU7La1klzw^m7gKm!aC-Hr~(17E{;A)58xu8~@n#+{tq7Y0~`wG*JtqsF}fu-kdeh?"
    "jX?XQ#6YSi2dYniqEzFoSlG1gUn}v!>fVTOyXLN9ALcdStM9YMB{`>cb*1@IP{$#Yn&<<hnf<|+g+2>6>?je#BjJ6>UefyEhgKhp"
    "-yL;(>F^5nhqRMT7NsVmx$<KCb*~!i77TGkj8iv=>CJ9*0(9QpoM9IgprD?Vvkk}JOj4M{#RPKGB~8Pai<mp`_GyAXp6*Ynix5UC"
    "`NKo@>Y;`8xj!>eg0jvPCPHk6=-"
    "7af_M2uUOof<GFxDh1XqZ}=}V<BCIQI3kW$cJiQ1D~7UG6a@p6FjqyW~#c5f|(a`xJWIreC*C-)Wa-wwfs-"
    "e&PzJ_Fbm^MTV=+mOy5j#Shq+0=y%g8dz{5bmgfIk>}wZ$04k%0?Lk8DgeC={HmhM{{hqfYkCNMaI@>!y<G^13;u<L^jgTSKd%Yt"
    "CT(Bl4E1Fc%hckBBp}ekuw$`@$t0Gj@bo4LZX9*=q_Po%M5VA1d8QK2Ay(Y*vbui(yyqy9GHYFTp7rTGh4!T0D5Y7lP_W4WAZsoG"
    "MG@gpRt$!$kWiO22itn)*=Hk+Omt*w8*F^Y_PQ}tv!?m`*<3nPld(}DTnP<(2|xM@?RN(WvaNS`w*xRSzf9IQMRxSvF|VsD<HPds"
    "5K`E08$=TcyNvGlpR84K(LJMBZgQ~_EpyhHkR2kOv=c~mfRwf!2q_)7Y;{2T)Gi&yz=e7&W=4rv)A;TkKCAaR41_QQqsyJ+S6|IL"
    "~V<G7{v$ht6od?<qk+Rj1o$%Qn$bvV%E*@)VP|T13HO@Kk(mH6l<V;5MvuBB_z<Uh3YWjR<eK@6iI6hpMCaz@@fE_@L=Yprm80f0"
    "nTNA+ne;dMx}(TlO}6%WIu3K_l-%8Q*?qb0&U(4i|BfE=-"
    "YxQ^Ug?{Zk{Ik=wYEuiGYBNpawyz&BV|>R{ZIhOR=+&fD4Gd@27`XB_hW}(BY9(Cz}HU!4{6kfJ5Q&(11|r>LW`Vh03~30&=|INt"
    "B>qU=IW|fiKS}13l%~8(Qf2Qtm;yiJH%4M^6Y8mIo9y$o?ang3vz~O<~VLtMi4POZU{lba_j-"
    "$wYgO5RT!(*RSQNFsbYVnF;JQ>IlU>&SX6vsJz@Dw)ol4q(u)hj?lCWsrAP+At*hO0Toq+@H^a2;6WvbbR(i!ij|oi8HcigV?)&S"
    "l_M6GzE~3w7Sgqdu>zPldGgS2bj)$i#BwM+dAiFo9p_&T>9n6J8ju(-"
    "lCR=<NEJ>Sp660@DYNFL=QSmb9TjVbD?MJ%!D1wSm@DJJ=^&8eUt09MS*QY;p|e~V$!oQJdONG2vEd~g##Q`s!WeiO&8v|vw9PQI"
    "J+tS@8Hb=_n5pW%PkEu>0?WtaA7O$*ZTlWZNDVzeu<@*!=3u}uBS}^Eo`Ul;aN*3!V#wt1dBzCSIY(6cAjM>C2fJ#QZt#^ig0{fQ"
    "s(%MvfG!Cbd>uh%v*?n@o62<6L3mm3L5ZBrcmgJN_Qbe6fV!)2llSO{{#E8tc5Ua{SL%H6^+E;|E|nqK*-"
    "YkT8q~T8K9B=UvMsu<)`$ivx0xi3I;oZM2=Xv^&)wpr@M@u}WH<ymPIAL>17Thx$85{gk=zPU5N^d+b~5BzI5^^S1F<_f5zE7@7M"
    "=T3Gy0eo!o4vS0eVZRU}68@`>n?_e*#tg?mJFGmm`>aKrIFvGi2zLln!9|GxQO|gpj;!P96NxOP4_jJ59*GU_<3e2KohX3@5SG>`"
    "a<W=J{hS72E{3j*lm;el`A%Ow0qN+MnTuEvD)+61#agQ#w*oJTEv0jIArB67mGfF}&=`e+nfNi}n77g<TJe?$cf_Z`DIbWMNY><h"
    "{A!6(#Veq<J0fb=D3=VzZPX4X)|$4CT`p$(MmAo>vHyi7n}wMzlu}@`aW9E|-"
    "0W5J=$clfv%t`NoOZ9q!yUc%x5zoS~TXLVZc?hwUb8X%vl8m@d?S47JT!i|ipL#v05khXw?a;FZs3OdMgsIDQbUzA|0EW~teLu?8"
    "iQ?qW5<84DRWl1qGxpd~tYdpWTU>j(}3dQ>GVvuZETn;OA>9mh{_5zDk4wgd++?gFO@-"
    "Hnyb@suNmSe%rtJEGui!jYa0YUL2aj?YBsm*72)5g#tBo;#fB4Xt9lc?P}rxbba2b2wAzN?y&r)9rqi`R%N+*#eYL&pp~4C4fi*5"
    "Dg%4#^yvIOP7S?nOwUZr(HCMvYg<U8b3T_uZz=PH<tn(k)Y?XhL@$y#aZK?Ban6;U^R3(x6SH$HJ^q{+6!%sXJ7ny!nmcYDA2RXM"
    "R^4p2Jxv(jqH0tQajlhE#U<d5#P3Fal%<-"
    "dSp?UtuR`w=#4O0z>}eY_c*eE#fg%X^5s{>U<+Ik?25Vik3PEMQ4u*_HtpErxRcOezpZArkyL*Cc;s*$tZ^71>ZzbT!_{XI3DXUp"
    "IUT7r@=|b##2nJJm!J*(?5RwzE3;Ar+Z3gUmF6wCzHRp~4|;7gt3yZzP+C2FGp#K^^NEe_LdDs(k&@wg?9nSLQPNGhH0cIsZ`*x7"
    "Q_K;u<vGo8V3-*wRi3TqqBl*@IUaViYIu=JHfFim@*wevC=CD?;o2nfM5%$J@^vdNMMT%qM40<Gc{-"
    "#{)4xncpYmFOMx^TuKUrTD9QX>sJNoDi{N}DQ6&7L{gH55jp7i==p>MXz3e*^QCMM^hp0B<$?l#X3`fr-"
    "blMNB94se5_Ib2k9NY=~FsC5i~9&c<PcktYwu4XKw-Duk?g=B^+PZBh3R6}ecOeVU)<!{N$X`o<~ThEr?CBQ>V2<r)sQIfmt2W3g"
    "vV}g5@a1bzX>7S-7l!e{1(3}uw+GMca@z?=~K_4+9K?CSV7k6;Z!Z)TOEH2uN!$B`t4Oh<z^jwQ(Tb$hNfHq7~0+7rAZ`mY18sL-"
    "b$2&$`!&t%-"
    "LyU7RlLI4QLw7M_8`#g8@#CCpzgePqXA;M$!&AgSUz6#n9<WN~SngHa?xEO)<z4RPup4kzqA{78Hmxg@itJwzc&jz^RIgylr%WU`"
    "HZX{R6>>`rOg%`Kk(^~$|C(Dm0J|m~L>rEOjlv(^_o`8pfacIIr%mK?uY1}Z2Afy|?$S9j{LF)%a}FK!U-"
    "L`4(Wz+T>U}@v&tM>2&MU*5egsUgyrSE{!G)*E#W-%3BRr@AE<M!$Q9qiF%(|P_H^<-"
    "7%w(G`Po4`DoXQ5#f{)eO=hQH+J!xZQ#RE8Ks{|XFM}s}SkQ@TxU&O{3^Lw$uhwc@b3BH0%mHwk82cAQyPsjfKriLDSLZcr<oR|a"
    "x9=kaJ&<2Wn>JeN2hs81Gn{@qhl0Z(g0PMuJbQZn^qlT}if0pOL@c>y2-+&7;;E|(16hYTUQkGRXG8-"
    "HM(HqlUi}l1=c08M$ZgYaIhzsyqnt<SOjRC4YIB#DGF*$?nNJBK~$;f=3kq)JYoUk&o82mF`jaED1m_G%1)6-"
    "Qm3Ovv7BLYSYG`LHu0GVyoU{N?CMjv4;=KPSp+`vTag40>%p-"
    "M@CW^lR&Rj+Em!_BNDJ2fJ6+uVQ)AdT*SR72M<&=l=2NBsf{QUY?YrB^h=Gl)NsL4GzRNZ!JHw&MU<L&NN(Hc9)+poM9fgE|6=Hn"
    "!z8Xn459J&?AF#puC+C@2{_Q#OV$DVTIdu>bWLx+hc7ZSd6P<S-"
    "s)W{}_Vku#E6iw>E*1S#Q%MsJjfSdTFi_8ULw_rlP@Iv);nJbSE14(}1^T2FUq%yfq-AwI~WrUiP&RBCy4Z;icKNMMkA(O`)D%_S"
    "JF_89)qtcyeTGnE%I-wgU1Y1M)~OwC?W5NZd~aFTE2m<8w(c3qm-"
    "*1ir29uzH8h>s`7e7ka=&)lcV6G^$bSX!=hj;cMhQGx(w?VZf#{W(Y@NQZ7;_7c`*oEg?#k<uag$JWEEsywZ5m8AdcoHqY@!@R&b"
    "slOkJX18+0^ZMVMY)ye>k~(f|JEGAr9rIeq1kENd2Z0R+4VZ%ENHAy;n2JF2{0)O*)D(Bli~Gt7X}M!+cH~Xt%x2fC<6hM$Zo#(w"
    "^?(X9(=C9b-"
    "r{#V=+#U*t#tYuAXw<QBAhVh(TXj5L`6ppH4P*sq}b~f0|Kam&zpcX<z;B+EoS)KmYfXwA%M_Os=z^h0~H`iVVekIo~9TewG>7$G"
    "LpExP31%(2#8*h(DMztdt20G5hRYa5$48o@cmU&Bg|gswCbe<GLdWBIKNi(wF2u{mlCxuO8QQ-JzWl>WAoYbnn%2L``saO2@nS#%"
    "{`5otv*A{J>M%i_48z41vr?=C-"
    "ZU?kuZnSiPZxzcBKB)Jh^yC4JN<J^tzmcheImM&w~HoTum3v^yV@72(*j$YI86^lt^xH%^T19Foxi!wr8CX;CfKwU*+5KEPv8{=Z"
    "pcG^kg15kj+=q;Q_^|Cb2au0qb@02l@BbIi_U^!^_amxhapd;UtPlgdgx;%5)9)fM)^8$xa;p9G8p<!@Yt}sA(3FZ(Ck-"
    "YbGaC{V0;q(Cs=;n&$*L`NMEOq9@qzSmAQ9N{5Gd-fCt_$n(nb7&6=gHS^4*RWxCnPeTd<Y%lpE;~qSFi}8Ksp46?auuLFFks(_|"
    "F5GhR>8KKkvm+#X>8IK7494sFdYs@ccmv%5Sry=S@(QNKR#^Q(hz(6XGE`a+<nJe^R{^@S`|u?!SC_?b1PFiFtCiyy>7gBOjZ{YX"
    "^Ke!7VyOR%1K1d1LJ@>dLG?dq^?H6NjJ!OFd-"
    "xH~>g^{f{_~I_5Qz~n13oU+#Q36N$`sr=Q=4vBj+Pk|UP`(iJiTtI!F-^>*tq}7_xgm>UUNIGpOUt-GC7>VpUQ*Wa~tS2V#b*X2v"
    "oss9N?=ReoVYXe+531CY3h!-"
    "~JiU!TA;0xVbRwgu<q#BX2vylI)BeK*h*B02xpIy{($*^mwLsGnWwW7QPv*qhZ7GbOl6hj=H(cn)8M}!Xo8KdfR$o7=V3H6Zm(#D"
    "7d%V|BJavIY2F>t!S{YnTpo0@rOR`;0tr_WbH{z<EZ5SB!KeSLIs*B%+XZa@6Y{lreiV>x$1Uc;$`}Vy^zwilk=DPZze59k=+^5W"
    "}6IRTFo|~IPWw{nbK3KPiMG2cWvXgo?JN+4^Lab_&9l7pr_XYYdp91Z*Ivo`_BGRmC{&x4r7-"
    "Rw%Taup0Nb5)ULRhbnIC1T*l2c2Z<7%ydP;fQ{a_Kfk~^8+AwVv{dyoSoJtN<c+e%9Op&}{rAy<_^Oq~Lx8o#pnLOjvE}i>l)r$8"
    "7YmgHr%#gcn#XL(noJ7ojU&BK$449gkZkHU!Mj&O#Eb;=XGDyW9xG@lf%A0iE-Yv`7OJEJS$kC3dS<OzCF{ZS{Cm|h=E_ZY#F-"
    "WgDa}i*?a;W`|8{?k5bqK&YqQZhej}O4_DS99cnidEZhenvU<9<&p;hbx~DZ+r4p0BX5T_~s?Cc$NH9DZ3fikD*4lZBZHQX>T6^P"
    "u`6iNPc;Vf|&PDH+);`<8IV^w1w<0%i6KtC)r_7JxlZb1d^g_j1N|PHeL)rnOoyqvK3@jN$S8QJRLc&R=7SbU}L#UT~~iToVhmw;"
    "V_H*^N`ri~or&&11Jp-wuHL`z2L4%*Ft^ne;|@ZTt$&w4U=>(PWe5Lqtz}dXdd95_B`hbBOZw402$z^nL}+#LRxu#U)h_iZ&Q4t#"
    "?!DGYds}=rQ$xIz2s(fnyNM4L96tO0QMJG|wPW*Boz-"
    "C}soxF&fi$F~)+vaomyfR2kt$v*5W|TDUy68MHz?_$~W@SWmX*kWS~drbNAFI}KuUw$EAQI;%Z+x(G04P~Jf0Uu8kh#SsJH%@8h^"
    "O=?GyAs8c(lsAha=yKYrB72L|eMH+&xL*>fp6A@bBxVXm>1GJ*D8az~ay(F2z?lk;3DSsju5+c(C74@E_c;fXBid(XE)4G9j<ydo"
    "59?*$1?M3Ado2s#sH{24kY?`+DQg-"
    "o@;qF~;1z4qV~oo{zKw_%hC}}qL?Ep|aJwcZG5*!QBHaVNYH01Sk@Uf9a(_848a62GF3|e<8@s))*~Swe*<Rz3NYI&P%pvMgxxo{"
    "p`Daf68u_QNvtjbHp@Z~>UyOdj%;YkMq|aZm;-)Lda`pk1`;lO$-"
    "|w$cN?te({xS?+;t)V4!5{&Ov{e!#g#N?|PfuzEUy@S8y!z7O3+^{0N1Bu`2Co8oZ={aKSPOY`N10CKoh=o(kfP9B`%>dMB9paX6"
    "U-(;ZuhNTx-D(Runio$nM7&0LARV9Ld~0zNo=c_f1w=L>H-AV)swX|4{dFf3AuAw7M|<D$zS6nw|%2M7|$attR5h-"
    "hbK&fJE2)T?DVd_7{7Sal#x*khsXv2D_KQ2MvU#c@|DgHaK(NF-v6iAa1im}F;QAZU_`3}=mpYvei{RsUmo-MzMNJ^tqjZn8-"
    "bLnRB-"
    "k(0cv<;h6!KTS{Ggg%(t?=w|BG0t&la;v?vDi7>u`JNjdgn6iS;o2$OM`I^}kIa+dWxQ{gBZZQGut0H{LEc0jiC%~>Lk6o4mjE8%"
    "_iadPrbJ95?ldz>pl6Np=-"
    "2cY5yNJ8fy@U=&L(x%Fv;Nf~mA&=kIZ{Ec5G@H+InpN|ht)aiodaJ4&JQI=Fi*xMPe)4*77I(jY&Znr~r{}vO+6hb&RFH8wOu~fW"
    "9^eaU*D@#ZL7qt9700S)dFPLfg@fYaAJKN+$m39)42z%a=34XS^kSh)Lw+dXy8Z+cTgJ{K+3vr5szpl@o3NWLP|tqu3{qjXX@Y-"
    "ffr(q&Bg1oHP&A<m9bQQBi1`5ODc2RjLu+VkhoL!mQ8gG8`$d=nIKtHNxE*CCV^e>sFC?H;QP}9wppEt^zJ+jTku>I7!z_S)CsKG"
    "v<mmU?MTjb+IVTQvnLdn~XW7Bt@rLuyBT%3wRsRIl%4el2Uz0{1FidPZ!XVI99w*y2&>UF^TV743Bd(DaLpsZ9goZ;8=ru(T_^8s"
    "y!S2I6^gM>4WT7Tdfte#*n8szu3TxP%1A`E{=~|{~L%rN<U?>bSNcd8-81)-"
    ";lb%9PT?aPm@B^{*K^2_2_O*)&*8Vh9%Agqj9$Y`HZdb7}AcO&jF|s3bEQ&1-Z(yYy?ZbR^+y-"
    "suo)i}9D*9TUKAt1NX@7S!Um>&=F1(FqQ|S1Z**}^e^1FcvT^ff|dNHwQd!R|VHD8<3;<lXkX#Oc~-"
    "j@eLBA+Pz^C;g$kzdUtzlgy+-{hx+!!?noi#C+5O&-"
    "ahiL^BS&_XYMXb`Ts#WJe<fa@3|G~7yygO+JISipl_{&cPgk4U8s31^yvZa3iUH%4uIYB(NKd^}(Dgk^CfFT_y{nOn~pm(2%?X%F"
    "P;k;p(sWg6IaObB(&us3VRFZRVIp-"
    "01XyJlnY$g*j{BAmvV;%1ML73G<Fq{XL(V0${m4!csur{>b96J}x))>6erv==L;4H!fefBW0v_jPbF>e1)Nyv##Fq1&g|?XZt9s{"
    "Xgr?vI2DXm9;1Hyof_NOZRKh>Xmx6`bbn&?U+-"
    "uE!NCi|Hkkr_siO(9J=T>QB;q2$5u70Ft$bp?D5+zwa$3019KqJ#?6Y*$1<osL=z6Y<ava_lzlsI45pJbMzlnr0M?^kBn9{1V4{c"
    "jM{<_4J#Q(K0x*#xfZ%G$*(;d0*r(S4h0jXYo$j-"
    "4A>b|{l%*C7*h|rTHAvvV~26RP#uDctTJH6KgIkHGrgla@Tg&OY_vY<mSTC;w!hs=>FbSuilM8$#t+?RsL)kNE<j%Mq8-BUk9U(-"
    "z*EG2n+iMO5k^YB={qr*wy?kF3=rTbJ!Y7hz&vi<5gyq$^+cG8AnQlhn&eT=>~-"
    "Bj0N1VO(`}k&<#aEE9<*_J%Tu0JpTuW6glw!Xvw*O}0X+~9#Z_bBJ;{|}05~u70=rI&sg#(Ij2dZ#z>5YTwd_h?6(&;Ayd6C22+D"
    "{{I<5Z+H@@ud=u`s>LjB;3r~?iG0wX|B6xqijFM-32N(aj9Ex&#!B0I;+ro_f69$ASwis@*!-"
    "L0IS>YT~4XA;pBqIRtwv+WvTk8$wT^nq>TCv%X=MuQE<a6rgEILoB{3lmHaR%iGjoMbg__s?VQBZzcTx|p|(=A8|brl}UXF+;(wt"
    "65PYb@6CSVn7YqrglTkY&%oB-8n=0GjEQl(5p=WPS5u+hO1nR1|t%WJys!n*Bo;chTFV;!1%}l=-"
    "2{`LqaR8ge78fI|mSYyGVb)XqcDji3vFxm(~DN)Pb0KmC~eu?nacV$^K3hP=HhA2}p5jyad=0Wks;hIGQk<06iSp91e#-"
    "9uyNqgNOm>gL$$?3cMVyrhxIeTd|WkP74&e>ohMms8VR(C^%q>pYdzWflGjpP^G4i{0hQ2cwi1P_}nx1P%Y*yR$G`+iNx#e<m>)s"
    "ZL&ZAm4_+l4{gktKt0QCczd$!@F>5wzH<2ENT3blkp@|!ZrjhI@YQ3FI^A+7qK2n$h06;_?XXVvmSPcU3rsl@{GAdw76%wP(tL<&"
    "nXn!6f$lNGFA`m<H1yyUqqFRbr^T8Lbkf|TLauF>&;g{&kP-rbN6opF5$Yn_gbyMx?15tCp%(`A28!5au{VN$dY&2O)r!|Cb)1Ql"
    "_tnBTT@$VmJGl)02!Hdlw$L4+nEC<JbiyeIUo2*s2TfzK052tt#DO*%VSs^K`EoI|XLvg<=r!<?Ey6G>BHs@Og{WX@73BDAit=~?"
    "&fp0bWx%yE)-"
    "4{ynx5<wR2<CY;|hV<m_zmk+JeQ>H|icaoFGImCY8;i0@y@uBSP6JI;Ugi2L#}euCqY8R1V8;>Sg`LP!9+Tola!A-"
    "!bPLuXNAsNOA!tu~55jb>5jG16PQVCg=|%qGFy5yYzU9+b*~CjfXFftU4+PPoU{{yifw|@t`*LyPYA}o3W<xWW$;QsoAPgZ_Pst$"
    "d-ztMaz|w40v+R$^q=qZ^{mhkLnbHof4H`tB)9jb!}a2w-uYA{FNLcyZAv8-et#pX*Rgt0n<vkyA6q8<-"
    "iAiNtJ+8{7914Pt`bxE1=1(+qyLPd6NSjd~H>wM7VqgC2h=;i8sF6_=4DaFUtV3m+9s%td%%~(D_bUmdqGSqc_t9;4@IZ6>1ImW?"
    "en>fc;f(mZTKv3C6Vj7EGZt8>%zIa99jA7VPu9w0?`?&L2ZAxeOeEp8o@So3)l~pR&ZDFg^BhFfBZo12m~QAv}C4k96c-7cbcd-"
    "j*1W=iK3Aau8Op?BinLuH=~NxV*{E^t1;Cq>wewYC&^+!CINWxjJ;Un52h^7WGReX$)L+*-"
    "B&$iad1*gp}r1YS>qMqI?T~<eYkX9cu<ii6?et9~LA8{~Qkb3<IbT!Sv`fX4jLBw}+GR;(rM}eNRL_oox`<I2Ih&MtZmr-LGb<(+"
    "03%>X-B0=$Hqh!8XfUgpNMSmt&WNbVp*Kn9W+*CdI<CV+-"
    "Bx<&o^w_`9Z=K*IVmN<;o)FPr_b%H|3W2#W1ziq1{iA&nk_1T^1OLpf9XLhjRoAFw@58Fe|*t(Ba|XURXP^n*_j-"
    "6oR41p<N5N*TM(m;qq&a#CT8-"
    "kVB@jbwU$S%m|OoQ$Pbs_|xntq53GOsuqLlV`zWDxZ}ihnR=l;a&9pSWbE}$M<LcD^54e9JBNXFOcLJO$&aFF`g*Q5dO1#082@yq"
    "v9qOgK}U3RQA!mc>*?EGyK1@WlAyPb#L$`;(3^Nem0vrIe+td7{TWmMzz>z54d;9>&<a2gi2d4(vK_Oczm;?!4rT(i5@=?W(NDB-"
    "^r7a=L?VC%-x!EKsHF<0xD9rB51;5U5if<_*T+MEZBtY@<U*s-"
    "Ab^Qr}bKL5y(PxI&lk#Hsr9!QAY^1zN!)r=UqpM_WgtlcIvai)C6?QV~4W{vre#}SF>(7As7n<J7C}9sTv$`a~R+-%MB7u+|p?7-"
    "vnYn3c^|*TdE8N4946Z4p?rY2wX2X2(G`hOaCg#?;u5?82ud`CWG4Be|h4ylP;h$xSrc%<ugCRKoz|)T(eUK%ov-Ju0D_ddF7x6g"
    "hfc-j?Bp~gpIO6mfJ4sD#k5L3E!Rr@V(++GHe}7W};Rx7-"
    "EDHgmpZF;!5E(V_abK?w=_rCEtQL>b`A$lc`E}F1|ssHFC~m!5ErkS$GbVINLO}7VXDuH!zP$=fCs2@78q0!2k%RNC1abc6mHYYI"
    "BfeB1`r};fr4{REG6q(G%}H#ru>8E=p9?rCwNnMuV8>hdhfi?A3@DMxNnsj5XfMn&R`gPbg0wQBvd0%!P(DM9G|Xb+8$1KLtnVrf"
    "bDTqCqB&Q&%xa;K;coq<^}+7sKdmp(D875>>1o51{~N;nM@xg7FQ>8q>pDLt!YmP_aKp%qoAdP{E|iiD}Jm+8FgwgJS4kxIWAd>@"
    "tApSo3jF-ACy2q_p$XU7F>L1fL0!J@V=0A*s#2;%=vx$Ocy0a)WaGz+;N#Yq#w$-Fx6o5u2GQ14I1h9N{stzUQ|JUpFIs7!2w;k-"
    "3i(@f@fzBCsK5?0v#XwdYK5E2Hf?#xR-abeJ2CLwTbv&4OprQll(Jd(cK{`0?B(FS%6rMdfl4gaDVbsi&76bKWtZwi!(Z4)%yA#N"
    "*XiW-"
    "<UG)*ke<UBKrBIXj8YtdURMMq>7MPZDV4k}td@Nmy2mcZ9;_brwn0Us<srj%V%io`YMDAgXzu0bRmg;Qq360^PEh`U6{d*)5pbv1"
    "ZQFyEl!$$K;YUtFp!<X3msB662&FWbtlpTd&Cn&zwH2rAB$G45KNerI!4x8feJ02s>=I8vn)%Limm^&MiwJ`vGAPN;#Utb{UP04F"
    "EM$Y)l$n%^h2OpGtP&(;fQWu$NhCh-"
    "@rsDlG0CB^GaR#BukXR)(ijSGPAN0}jBUi)~643?rk5(4%ORi~wjchdSAi5Ru_ZgCiFr&xBZ7x$GAJ#@yIG#xFh;oR6u!@OHQR&2"
    "vjMl;tQ)Xsl5!<z|DO!=x@+O*NGVKTjB4=g(AMLL|gA=3(8~=+KmbK=!}!orj(Uy^%zJd*oJB?KT!5jqqITB*!z#8|@AR^?j0hVh"
    "Dr$#xv*&+XjDN)G=H5WTUe^PY43$1t$MZNX2vBR*LI4;LqNT@R<*dmrC^T%YZG0Z);K(2Gkem4k%yaGD$kah-"
    "{Ygh;`Bwjl;UhnFGV7Aqgm_JtGfplXz`<1X>w^TdnNbOwV+Vnem8*j)x>fHI>XmCb<F=PRzq=_>PC&v$L%+u93)Y(7}E1$g~_UjX"
    "2=>o#*1BMS%m5kC~XV*DT<i7%Cmg>!cV=C-uxiVxDcW35ZkElu5f7)u9i9PpSNsfXILY{8K>lD*f3f-"
    "2!Bip|XHHKogsmBYd5*{c}NqN@BGHBdBaK!-ArZm~STE=CiPKIl1fiH&_jVKWBbOaT}5&ye&A{>|q-"
    "&;Luj6N}esse?{7`^<>ygM6+?BAg0}ohfy3C^9z{?-G}NHQoew#GJm%oTJvIT;SO5P@hlYP@pam6I3g$i9o`5%r-"
    "^Vd?RPh)*SOB3l%I^63J=?!-pq3lDIq<~XzEn#Axiq{!^Mpf3<~nd*g+#ji@ef2g&GFRCn-"
    "88b*0`9rXs~2hX1i*F&Sv^jJkalilqfzlnioE+|{%klw8LydaGvWc*Y${PrUv{E?C@{sDsJ<--"
    "*#oU~bIJp*yteF=6#&MK(*U=bEEl#Q~B=_-~Tm3bWDSp&zUG2UK$&hKy<>Cp!*vIiYxQ=_6I*3K2Yd_`@|$Z@Fa&-"
    "@rC{@s3~+LR^C*wi4ln&4@qMza3q&Jm-cwSd~J}o9XFl2arY$Q9-h~ac-hl#y!q>Z0z3oP0nnVd6LsW0Z!d1<CP&WGWTSx($G;F!"
    "%iHfgC(y-nng1Toko(I7TG$m22Jz($Gim(5}J`CpbuLvd_DwPj{L%Vij<gN%TbIi1_UoZ>o-"
    "gO&pSMHC|Db5BM!arnd%CYIZILIdcxB0u?@29L{ElvYFJ_rnouI%#_7@B%d+yug2?008M+rJnRP!9;t)ZE{Xe`K+CfNVeC)^8d9+"
    "bFe$T(*62tn)25W(_Qlln3;^BmO&-WWhjhT%8a@oyA_Y}t*Bb_ej6`|)#-r=*e@-"
    "}BF+vA93p}`&hyQD{AM9HZS@=$y*SmL95%G|l>Q#CwF;CW&vU5J^UB_5PZ_;&jESGqOR$mI0Oai}~D_rn+lp&b=v0OWrV>xo#hoV"
    ">eT&DgRwUP|@~i3izB_l%WNkU<i;p3jwZXOB~)4&sPR(hnOW+^Y?yr74T2{T$0+QkF<p1^wMN{=O4DfVT@=*TWZdCrPY5k0qwHn4"
    "^85%sx*ry>zZ$eobS@7yyv1DB|(3?CI~rhRsUZLx6Czz*pkc7-"
    "Jkq7<&yRC&%IuWN%16@v5ppWh~lZ@6mG$4|5x5hg|u7Io`2*{xM@(UYhhysxjjX9W%dDDS$j<%m~pmLGa_QJpBf8%WQA=0D)np*m"
    "SA%+30FvgHXG-k^vsu(;%+B2~AR24M5wAmOHi`gB>gcZ8I!Rarj-<ahf-iZvn$aOg{0h_!9QQDRo)j?o=b6&N5qOp$$u0dSs!AHn"
    "a1ikV4}Jfa)2vS$ELB#}J*gwmNP^y=IaymP?TWHd|>R5%>%)q#qRHG`p2aHy984_3mL7(<;Y&&rX*(`BjdkMJTG$ZW_Q#hUxUoQG"
    "6|#tyambInUt4RG{I}Gfga7wvgC4?a9JwR5TdpGNdO;N6d=wg~P=m(jx3~Jb;CU+!yU}>(T&XIp{c36Goz!h`Q^1LH7!@48Umpo7"
    "u7?>o7z-"
    "Q#9mdWQ?%QEvtq(ofs|}5&J`Nn#Bc?T42w_IOQDLNTK{e>jkXJD1pwS!4SrW)2LQd3s@l@Yh!HX>2eP9K`hiEudKqam{nlgNr<#P"
    "Robk@2@%N1R_q~!2|M~SWu*zKPg4See0m5aJcpVD_|gDR3r#0GwRpA%`N$s9oXpH8Z}(?(ZFdBd{L<tbN-"
    "O!PCkd?ZZ^a{>l*2RYidMtOU5Mq(-EKAN9IU-DJ~JKSx~+7_^UOG00Bb`xW(ZMg0J%lO2g*3*6=bHl(@uiI<ax;2`^xIXoLX8-"
    "f$`)hLn`25`#@mbWXm^8GtnEcZ_lqIhl&e+hg}=9@rR!hqP~GY>HeY2Fxl*xRU?%L0LckS$*2{%b{MB#Z}3t849j+{S4M96S01m;"
    "7JGQ1zuedh7?4r|JBNrKq+!QmE9QF^0tO87Y}$<i`=`=CJd9@+4aVdnVc16p9s<<PK|0;O8{0nX=VxM0(1CM817T+S4g1;yArqg>"
    "a;{<gk4*`miN}99&iFfEDn>&z8yGWY7h$RUc8;st?$klgKQ%QN5+6wu9nn!B0qrvk{M}Saq8#Tk(5I)R?kXL?Hk2F!x`3(_qS~+#"
    "2LyoW`B!oFmq#no2FxjEz`<Z1&*POtS%dk{3eMt}Gr~R202v_vua`?aO#nS`C{w(p_|0jvXDw-"
    "<?^8}xSPJ4|drW3MIUC3wSkr&Wns%93Alafb!5tTVhKrzWtnk-"
    "J#U<Sy`=&FKGrc>a$44N`Vc#Cw>>Atquk0pJpmnDq)ntK}dQ9fqj7;6xODC}>X`jdwgVx1>Z|VlJv~1vRgA2$g$)Xc1#j*>2KS0F"
    "#(v#YAEIW77b3zSV(J#yaS{dU<1<h%PN-"
    "S9hgHg`Ij$p5*`qE`;0EqO~b9fy_7z~#4RKzn!@d`Z#6F3~MQ(4~;;yI(S{rZaF$zU~#N7F!d8PAeP!5&w5N)I=E4>|!Fk(e_j+("
    "44W5#MY&6nS>HZ}zog_st%kqqW_*u>8$&op3WetqCS{YB{LP(Y2aIFiY;smtd^=8H>xz6oFfeB5}{VfNT~BU6);xN*PB4ijCINUA"
    "n9=smGWohr80sal7O{t}xKiSLFw#8S=36{8ZV1N#F^~6G^Kj&eZpL2Ejf*4m42>P@FkF!SI^PB2tuVuk0qB0iueS@Xzxak;#XL4e"
    "Zq@f?xy;EOnbTD;cWLE#feZbh+oC?+a&)GQ^g-)yt+@x6ozo9pRXK7V*jQv3$!CnE$0&!jVwZv*~y-"
    "*TF$J8!9kL!J*tY93{$yQ;TDa_4hZF+!P1)i{VyBs=J}UL!NWjyvTpnfFcBeKmHZN$KD5wpM+papYu;kj@tjT!^*3n#4%JDI>|S4"
    "C<gpiHdRc!`+n2&Z2p9T)4sTgUSjivQKY%1{b$0@m7{h<wNNB5I?VTTNp0WUWS-dJ6dq!^>@g|Vx|3~7>I6j=2qw%I>YQ8Z>EW9t"
    "{lde!rjMccvaMD>^JfoEng-%ADS9hGN1TTh7qf0+0;EUOVv<Rcp2Y$hU<-"
    "4$F}q@u(9C8kG5;lfCYK$D8RbZyrw}4a%FC?bm2SE$x#q!$cl`#WA2-"
    "6VE5SFmI$0;g4@A<*p!M|e_k46Zgp~fNq_fBI=SdVbCYijS!w7!Wm(0mYA0K(iPEPx0_t^=54bO(>q)!>#=ZQ^Z7Ne3MdPg%3Y`i"
    "6+(8Vd3ve+9e9h#w$+HvzzjI1bDw!u3sN~T}_98j=3A;)RgCreM8u5BMZK(8{L{+OVAOp?-"
    "LFBg<ZyO53FY0hu<bpczp&!3s7=4b7vQE()~FZGwE5x{$Co?bY!&u$fAgOL7G14Kx2Ut#j$@hX6eoVnws)nbl1A*Y2lx0@%Hi64r"
    "E<(AQ{v1_<@Q=~!GHBK6YYUEIGwK?K*0*50fXECQD=|_^waK>b!$}oXA4=Y-"
    "0z_Rv3{`rD|2m*$|NdGH6b>Rlu7>E=6uT<|HBTx5uNcUeUDqz02E#``O3c<!SFT`LoYaBAUfF!}~!%6VPBv#fjJ*gxTvKyJDL$Q}"
    "2C5&^}Dc6H(%DD{AE3hZiuTIWAr^qI^9~PRGbc$)8;NFu&zD)e4=hu($zx(+>(crw1vil%_mb=KfH!amvh%<}Tb6p@}C9wj3GMaG"
    "4H-6YZ{yJDDGo>o)lok2?qzyE`lpo_c)Hzo+4=!iL!~JHC0@g>ZN`2bAtTK-ujWd6d%b_>mBF2xq54?(!{QHA=(4%Ek@1>>xC3%U"
    "JiV6{VOD!tQ8qornWc%e%Ra1u^(chBM<XAP+HW^r8PIAB=Kr3$6+%FlT7$YP6AW)oI=ES{aukew*`!FM>8wHDn(b>_%;V1=QXK2q"
    "TCmWm_IOH3_E9D8>vP?jO=Ya|VNs?i5Kq41_1s(C5G?D6timVmb!;7R7z%a#F<&b|o6p;y&oU}>H<epF#vB_IH9d*<58snu}X{jz"
    "R^|R<bQUT0ApqtRtp7;(164Na!-"
    "!{a~ac@Rf82!Ll*sL3jGeX10XnK|Z?ZYmD=>YFvm1T6SoAsU?DshR=Gsw|#&6*(&hpAcrO_k8bk0uIAiiLuAKs10nbn#J@{aVybj"
    "{vgom%iks`$Hb(ajkBd`0U_O+IWKk|D>xVsz|s$2>r5m#3@g(d`1M~CD`%O!>s_4Zu<Bm+hTw=_Rnz$;)H?m`=Be8UqXKc$9#akn"
    "k{a_2S$VrJkzrT@{)7b>9aU<9IM$4NfH7RV_7lWoq4tfRpZqHD!BFor`<D0eL~2_`_VBYqOm#3o_R`4FCR&UilE8@l+AgUR@EK|a"
    "ck7j-"
    "v(uxQCTRLD^VvtBRT~9DbE)2D~=^PF|+1IB6p{;s7UPEKNCb)Zt$R>g3O)n5%^7gHZ$_nw)EhEyD?(^dtxHjqj2Yp$;35~kx}<}C"
    "w<SPnIgPc)`0)>FbtXvx9;It5OH2A`tX&x9+%u1NSOBJ!`x9^A#)BP+FPhW2s`)ubB$OpdV+h7eeetbQve9OVaPxOgVyV_bTv;GP"
    "u9K6hJAk?r9JD(UlY?!(8)(7662G*+Z}EjJc17NQJP4C6{#VDs06ixN(E~NoH7+2o}FwpbU756p3^*XqU|uE_J|n{{$=~W8fEAC@"
    "gOERla~hERx5ZZVKx$r3hbH4Gu%-"
    "9$(A22e@~$0kH)X<XxBoF^h$E_Y=3R*ADqE+mneZ?t{`079ql(Z3=5#U1PdmDn|qI=0lq{0UMDTYnFvG>XPoT4zzsbVpF5j5cpM<"
    "y=yoiOa^u>h(rOMm=q3H#bV?z^aE{fSaXp-"
    "cLkNDLW!-C=Z#X95{TI~9^Wy!r?>F2XpE4vG^cgXl&6$LLw>WTkH)Wdm*&dk#BP}+dyxU>hqv-PYhV35A>tc#tZcpSJijI@bh)Su"
    "j=*H|vWWSyjN%^+^oxPSx3_VPE{=?rTgSW?0LsB=B8)4=N^ER59dYL;gz1Up?RH+%mxIBAj9#8y=L;0A(i)%Sn6`7FYrU~X}C_aa"
    "6A2Yx|Pf`Ag-EB~vox?$$fURE`v*1CD(li9J6mcIj$Sg&>ISdTl^HheLG;H!{>z!%al`ZRn@kTQT>?ck^k5Psc{$Ziha|CgZ$(Yw"
    "%<*4Kd_l$-|y;~%}9s+<t!`?OSPY=&WFziX+nS@0X{TYDF(CXZJ2FPK4F$nmpCdc#o0!Kxnmj@IVA2T_>^zro^m1E&DC-"
    "{%nK(80bE30jY1J{+0#qD3AP{$P2=7G^CoteAsZ0NU;NJU-`M2e#z{s%R-"
    "3WLPwiApXvlDEwC*6VIHhe~$2k1dq&44CDH&A27=D{+}vY0-"
    "GhN<Z$uz4R?1Tl!3@#Z>U^nxSE{X!2Z|dm)zr^`qr6E&i;D;BEsSw$`q*?J^%`{2zf4E3<cPe@gCO$QJvSMljTgIF^;79ppU28F@"
    "aUoaxd%4Yy3QJovtm+VGWM!q(2EOwvGplxP>$63aI1U;Z(DJc!iTm(vB!5#=mpl5U1vC1CHlD&5m{amy=m?X|ro<vBK?<)s3XYT2"
    "Q3R<mPyjGj|Z#?Xv(76U|D=B69dCB*=FKvIt<NiNEm9Y}oXBicpuJ<0Trr*~$wY-HxtQ_WuWd6AG(sjYxJzq)aP-1byS%gj>5x-"
    "9MfBICQ+aYSB;;2$r%Q#rqFblj-"
    "E7@(w7XLzoA6lulYC7|K{5meG(tHKdP9_sc$va@K+08ldhWQCO9qdyR3;s8^<GhkWBzrcoJ4Fs+ZP=0&3O}?LDExFd}Hnz?pfR?D"
    "cs_QrMBF4`1uW&}KP61up=*0lu1=0(dl_bpbjPl)(XVdvN8#nR}l5l`cXd^q`G=d~psy7soDWrRLtfU4t8$7<lWP8AAkd2%eW)D#"
    "yA&ei&smhE!+i3KfJkt?Ue4d&B?Ka0N;af`GPV+d5O-xac){%*k_+-"
    "m?25EPA(qI&9lu(r0OuyXeiC}%3#hA>H$y0IL`uW);f4ritVp_(8K%bP~pK)3(L?HT}MCR<AL4Rtr{%7onRX|KC#XtWC|HJ#?JfX"
    "L|{v;!~h(rvE_zo-wnoIa>=><mSr6*yR_DXJ{NF8^$IHTeWvPzJazbBrJ68+eYo#)d_b7XfGB&A3kzg<osE4ap3Se3mknmX7|IM}"
    "gy(ryiTGEr*S9r0vU0c9PY#VT86Z~`~r=r@33)|6B<YZ!RqAe7BckIDmq>{SGQma{B+Tuf7qH3`R%5@|A$sg5hAsvN7-"
    "keltFjGdHT`vYx4dD%cKrK;>6LY5*1_O!<lU(*0~63pyKXO=cXjI6O>0W;g>Sz{f{JhJBL0pBgT?b8vyp+A_MSeaLm&oLl`g|Rb3"
    "dB~Me?-"
    "9giY$)#1wzf!ovRbg=HsA!+Rd1#K(gH9Olv$#olYmO@q@CHX<wpndP6O~+9vUqdkMB90cqCCZ6AQZvAWQgztzN~p9BkUMqikfE=t"
    "AHBf>O*Jc7<vj%Rk!Tg(ZAeCgAOZv(hf(;qBqy&xB!=ahdeA9OuD_mE1>aN`}03_?&R05<sV7z;SNgpR|f{dZU!}e31yBAlpn!fj"
    "t`ACuF4$G9=Qca^~!I7m4k2`;VbITteVBSVt%{h3>{Lo}cCP;chq6DLuWhs1*K*!(K0Z4W-"
    "xeblZbX#rvwXWhN7jJCMe5z}2?uv%3ssib`r$5vL+t*{wPp&*Q9Mcapxnl=Tuk5xNb}KTCL*hL>3%JIqThjTxTK3z;-"
    "2pUqNLH;inmmZv&-"
    "0LMiP75yungW;QDat1#>=W;=(MvT^>6DB8#9MYWmRrn)}P_Qf$O$ym#b{Y&9_(iSEdvO=VgX{U;Gu<OE2Nr>+kA|&=9MD7=#(jHb"
    "_*mq(VI{%=pD$KCj&nA_A|hoA2*=iC$58ijejGIP-|)tEx_G(=zcInR_CJZDfwYbspxG`viaLBhJj{&d?5}DjE<byl`#--"
    "?9zo!k*B;WdS_%}{!@WhEL9{#Ku$y*RaiTQtPQx6(UP|Hg=8p*#Gz+LcIpc$h#w6#4NwMV*8QqO;L*eyOid>}`>9V?VjBH#qbEca"
    "OvO-"
    "4!AYjpF3oC)@{6UT00n24NrLc^DyL%O{*hJ(|F`%M*Kywy<MC^?n49*5APK4r3ootJmlvmkOhULu+<2aO13@~jG{3GY`Oph+Y2KT"
    "IV_=fHGoT0nGjwp(K9fqb00(j0r^<)$pkoFmi6-"
    "<9%cz`C1!L(mTi(;;gY1eT~yf2hFNbScfS7c_<Sg{9*vxfUo7Eo9)vwWua{PrHR#I;)7c&_3mMYHmi3>9f9om1WzYW2VwwvED3_m"
    "hak@YjBwMox#qI1$h=Qq|&$(Owy?OswoZsr+UCF@R(^o&X)$<gu=B0@?t@<!}vK-LQa`4MpN-Zp;FV_Ai4UUOaf?c0j+Mo78qW#e"
    "f;yHZu+kk2iqMLBlycF|f}D(}HXb0VohHjz6r$Id*2bY&1-gpIcEc0~oTnYg1114I`*=K~$)W^!m-TqI<_-"
    "1>*94j*}41Y;+8&kg`T_2UruEiB8q-#vusc6iQe&B|V6SgWUnN=sEOxO!P;ioSv#&A($k%07-Aqnh@m-=f5w9t#Qi2-"
    "f??u)>5NF>1HaxuDi(|XR7J2FEO#btP1Q^e{hH(i!NekH_L#53@w^E7+|4d85S??R1>3;JDx{o4ONb;r~8Js<-"
    "`c0z;Iqn`!FXy2$BR{awdjk$B5=sId#wMLY$U=eGSk~U=lol*3STlI^a+?)Z77ks1wo||M^xHysk<IH3qJ?sZ)p4Lc^U?|10VN{W"
    "9PY8*wBF93^3MzKX2pmeL5nAng(8(V0}xA_`>*XJ>h4!?*wN_oHceZovYJ0SS6%<gz`WS{gMtjV@r>RAYV)&4^aJ3%HxyK12$po}"
    "HHsDfC!os(vPjM%?E7;Z0ySU88-"
    "WKe3zXb7YC(Nqi7%`17*$?W7!_f`Fho*cy(iJ;0Ew?y?m}Pvj~A?sM!j*eZJBl;QTHjx}h4AsS6&tJ(ocEeR?Z9Z|wwN{b=BQpv)"
    ">fCruoC*$|<P%w!aIlBqW42BGf%DI2DH0>6^OoW%ysjO8wc>Dp{WP^pCRwOkm?Por_@Oh*guELRsH4kWFJfzKkPQsn^M4jHX(&>4"
    "2*RU-Zcx8lE2=JJkG(V+m=57l(nc&cO&ZS29t9R4vU6xlv>Ht(Ug7HNx41pxh#}Abht)N}fMOry<LE(dxUX^Auz7-"
    "a8)kd@Km|7LH=2x=X4$GIdiK%-"
    "3Sxg}`f;3p;DDfEkH&)4imq`YJ5H3kZEsuxA*3pMZ&$Nvcon|@l6yaBr8A#x!kLF{~mTdA_HTn?!^qfOW2U_74C!tq!#!<>!vk%l"
    "-wC;&s$NmyfX0tLh#S;=D$!re5z@&k#h;_vK86F@hhbq%ew+(eZi~*crpdlvB95Wp72A<x@3O$|sn=_r4*&=%w%UMrSx{0=LJE!p"
    "KQ$us?EJ#!0%M+JO1$Vs5sl$iXz;oK6G^qbDJKdI!A+)e5@-"
    "Z}%<ynSUQ<e7tB@sdi3_L2+h>0_>M^SLTEFXC`RLGl_XVQBR#~9SAkU=;^`%cR_2~lBihc^pJKHb5jlhD`%`ZHPM)|m@a1J&!Z&M"
    ")#I8fSw!zxLP8q8W;>8z3;NV=pSx(>2=>4x)S(dU(V$g{PUDA09c)Gc%x2A}A~O8meVh#H6Lwe;N~vqd46-"
    "vg5n4*R*{t&oXTl7j+(EM`?W(9z%oDL<o$CqXx1l;JORI8J&5Gss5DXz##c<J(mbd5#u|57GoIrx)A!~0fD;Pql=Jb?qq_R6uVJn"
    "hI}e9jf)8egM2Y251-o@#|D<j8p;d;St?xx8+PAueips-"
    "9&WU^0uY8G3*`b^j}~m#4#r9a0JN}^3WFrC*>z~tBs@B~!x}|qe<z0m+ZJe-"
    "!$8?rjiwk+H|!cVVxkqPdGo9WOs`6xYW07&&GJs`W@KhWT|8R%tUgt@ZUyI25~V=k8F_`*KG$$^DjCStb&7>iQs%1NC<%4sD-Rdu"
    "TRM}n4}SaEEF&*NTn!?W_#2Wl+esN;zUj(kVL4N=PS4DeoiQ!%v<di(4pO(JrCH{q{277o8QL3Kr%@xppJ%1IuhmkE0SFIg4ip#v"
    "-ST&UQFm<p@Re^W6&%m**-"
    "SD?&yzuI&#q5!%H!OEjE<dBx1V1&o|Q;p&hkKTXPmzPe+wEq1BlXvA1gd8pP9fs%gPsK6k5Z^?A88r-?SKbT1QzLXF5A4_-"
    "<%oezRMLJyAW@xaV*uBK#vVLXg<iabzJgM-rC%oB5mYz?vuSB7lo0w|I*P#5`U;3lA%=H$y89d!WeB%;Z1|3LD|nMrC&$kB-Ijy_"
    "%!Uh_=WdY`cmAwuRfhEWDK&s|P<cQ&`~YrBCp)-<OrV__`E+$n&?i$-"
    "J+4AB45RJP+v8$;M|qCSBmgsqhl_r#vuoz^f;k@$LNoblq95Bw28r;l1#=NyI#8D+FHzXbpN0w0Gra?jBVnL<kbZW_8!SnGxssVD"
    "?);l?gO0BUGp>y5V&x9Dy0)V|IcP_?8ZoG3F?Zj=_IdzYtV=Qm`FAt)&i)SwAVeaOpnWbo0rhZuwKdqP%8Dl)A3}vri}Z7VvyN;d"
    "@9tiC~by$zz@)FJ74&A7KqMtLHGR!pTe{9GS3W<)R0j^?CnFiaUxfNBumTad2{UJt0=KOxFll=bWg^qvr~q|0ROl987=7Vk4TgEM"
    "LdNX9u(<LS!G!F;N6#d$EN|*I?LY+a8(M=&gBI(0M4C8j#73XtQCgv6>e3`j{q`Z0&v9hCmy|%(~(rU&G%ybLk@y0~}9$5O;yl8;"
    "+D?pL5g>-MHd-"
    "j*kmU4)`Qm;jFU4kR|zy5dn<<hJz)552K)ZgvcxC&SQ1_@~Dy|!`X0lZWd&p+2|`VZwO6_p=3rUCa9fEkK_pcK)|tU{LB(7D(c}G"
    "A*G^#rjTs^d}n0;Y)PR@I{elESj(J*PujK3l$noSrQoC3_vS<ph~qqv3wBzh0YeEI(o){zL=lIk_Hh#oz?d-owTZZ5`<?jNlj-"
    "!ZdLFo-1TeW%?%m&)I4W_X1-"
    "2h;YJ2fm&owK|Z<T<!q|Ko7qyg_anJR%s5oJ*)Qdy^Kkg0NRBLeQjW>d^;fljS$I@uB>OD*NapR`g=CwCV4`Eh-xY6J{-"
    ")Je%9?P89d?Jm&6!~`w&Yq?I6OG01J!E9@Y+=jhPLgXyARoX43JC7w;x;rY%)nu=ajXf9_0KccIeDe#EiZkS!624m}g{qtcjLGs<"
    "oRP^g){h8~s@~AiI5=R-"
    "He@@JC8=}X^qATQrYG_!KXX9XGMa;mD7t#JXaDtvnL}4rixR&aPO9>|lwykK6T(AvI}!Fqq**A&v)(s}Bq(<XB+SwuIf0l7;~mOx"
    "4f<H%wovvk^;(vvsw3KloI9UckUC_^8vNP2w*aJRd`DIi(~Oc&+e;h)3^X?;$%jWQXh&p)U4&9!d5jWQw)a6|yCur)i*H|Z5$uFS"
    "xC%}7I0NcJ2@RX43X4K4YwKP3S0|AD?oEBX4frI&h9Rs-NXxhPcWcSF$0YZAGq&OFn5=_L7*qd#h#t5*JrCOY)Bk()IyF%}y3g~="
    "KZ``diDFmUx!zk5-"
    "I6)ji<TsbpJ|ArkNL8BpXAQF)7~#>(5}RAkQM!DYs!{5bXI|Nozukjn&QNg|M%DTXzfiH0#7XXUdJjaUZ|~sYkK{zrbhvj4D|a1M"
    "|v005!=+ejT^&A;Le+Sw^-XT@0#9225OjkfJ+Le;@gz9-"
    "a70^B*`6HQ*E741UyB?4~=|U@;oQu*pR;|$kNBu3LvOg)6j&o+?gEmdCgES<D%%-"
    "2<c0L?J1`bIa7_E&Dsh8e24nqnlfg>ta!%>SkFl+A$<%9*!g|dA0;Dnmw-y2qrcUaAAuT)hv~0T!3(&^0D-d6jj9Ss-"
    "Eqs{z9-nrZ%M+IrF#8L6p7SQ8S(evrL(qRH_q+DN0O6CDpVpLGr-HL>1C0yT%Ytc1822gOb}CltXIV7VZn9XN3egV+KsTh?oKj~D"
    "4J-MO3{v-fUAdz!x1qSF!a{MI3$9wQG3bZ(}CDHx6k<u!B?{9-"
    "ohn_QFt*+tWj!S36DS7T^JK;&zOBDpmAi<YqLs7M>#Y|_pXkh;HmxHW*9QL<2z!_PUnntaDXf?4fW2%@_t#kPpY0@XUl5X`7Vrc*"
    "d!LgT@G2#yn*y7w8yh(!twie7dYxKab5qn#Sf}#p6hWhq;81#OAiFE5Fa=>&S3rO8FJ9yWo<zXs9olg#8cZ)hSPn*A;Qo!_2RLtx"
    "!5KUgYsAtzK*d4SA5ZDd$ViH6t~?Iu%W(wOd3j}v;*shMP*WO-~pm-Pp^Uf>I^h-Qq>{I8E2$-grzhls-4Qq-*O+LrCO3mpNx-"
    "ita~+=A?O)rTE`-jCY)yyZsmekFciV4=g8-9bXblA8?E^DG&wtUpWl~lZ+S3}x+(^9KcZ89YW6HL-jd=Keb-((XN5%!I-"
    "vFUY|H|03leLeH-r#bK-+rEie3A%Im)+i;W~*>Q!e5i+vMpu7*j`6k$e$L{VL%HSS=;xxF?52e0>9M{d2OAE^Y-"
    ";_~6`z=$0{nRBMw^ML3G_q3}1W>2qn19X(vJniha;CA}p6r{S(VaRZ?DZL^kJQ@Lk^hBQ3<!6QZ*ZlFv3((|`md8>k#(5=qdM9SY"
    "7Y3syMWk7hu5(ZH*dwKm&=GgZ8?S)TbxuMW{Hgw8-"
    "n6c7Yez;t=<)FI$Bi16VoP6W6@p?w9QwPWvzEiYr$4F=Z?SoWDa~N!}*3$3-"
    "e!tX6l%p&uO4YS3q)?euVFvck+p9aVtViZw?<e;%!q)RMeO?6uUK3}eeOGZ8U)uf(wgkf~A+6i~XZM^@jC1AaZDROaJMz8UipePU"
    "sO7;(I8CRVF;*wz$2zoo@KW>0b}`J_IwBa;#cWc`*@tYhbSQUT57*fNkx|X%P)wfC$iiJDYv`<&kWKs2;@k9D|4lsH8$ODUU**;h"
    "VW{az(oD(Lq7%=M*{a#&Hs_6k%gfF2GKy8J2uP@x?Brt)I;!p4{>Y4?4opa*gqev*VhA-"
    "J<I$fimE^MRH0EctO`Vo=)TaZ`w~hWB!#VMWSF*g$*@3YfvrO>%x3nY=h$SbG%cP|IE*<8UtsE$8J;#(;=9>p5z%=c4Y@QT<-"
    "1AuK3;b(KY-"
    "7%zT{6in)Z+Pl`dQKnM889Xv_3756exry=7b%fmL6WufgIV3PcRzcdPZG$hh3jpoEvo8`AmSe)?=fbH&8mi|75~6jh77#xBt#cwF"
    "P;YH~nCkt3QY!^dxtzK8uvP4)koc&IV?@cqeb2%iEA8i5lo&k3@q`YHX0bJY-"
    "o0&x6TkbVM$^k16P`D8n<DtG6?;4d*ENqEqZ+w9M)D<~Vc{46=&*u+BXDAk$2WPgDEvEqpgRj2zUHU!GPftSycz6t6bm<TmCCy>Z"
    "UQ-h0NHLW|7p^-"
    "KZib`+X;B?!Jj9OOk$7}m$x`_^b@oYTsm4&9XZ<^oSxXmM^t_WBaHl+SIGO?)+shkL8LNix3V_c54UBMUY(0JX}}e;JO~Tcc~e|J"
    "hF7r~n)-zpt@CUPRw+wLOfSc5ITn+APr-oISl#MEE^eE%px$zh{<gk;&}WQy}<YTUKF|qfmn#p=WgFTxV~SL7yoys8M_6b5`UZBc"
    "$8*Y(BJ^Bx~n}vbatQk;#*ZC2qw%Vc2)dRGmcbs&9zp-"
    "}Gz0>liW=2sc>BoBn2Uv;uy#Kv7@xCwyfTe1afNP!CmC(x{CGV=$=w=;og|@61~I6Xnea?<Irt5-"
    "jk1#~Sm5&NMDU9;GKB49vBjM3EJ%@afcMzrX)o(aulrvj2VG)Vj20y>ZicalK>$GWb_-XJ!^F`Puo=qL9J~#P5EvDUW`4T5z8{<#"
    "$WyO&%S!21L!NG|!eidsyY)GYQ?Ie%lhrFpIe@iODDIZ|~(a`H#XR8JIn9Wg3d)fHbY!b?OhJOgn|OBVF9B7QP8t()ZRs3(?iVcr"
    "T-sWQj<WET+To>-82wfq@c=$-btiaU$4LZWFGc8v31JXF88Qh8~+uiVyq#W6)1dC26WUik6HY5kdu+_*MiY2?)fXM`JA-"
    "6a)MgLJvmW?*VDm<Ql!Xqf+MbM;ZM6NkYZVigo63R?HqvGvv90UkwD{B7x^M602VW#BRgO!gA4F8S@==VsL@IN{R?B{_U;v%BkuV"
    "$yv&yURRT7Q)Vlob!bW5oi$V?0==X7Jd(JKR_+XKG(9kt0r{hEn8+C37|J-"
    ")w*B*LCGCU^Kd{a^SA9I&5}8L?rpcMPziC)G)*AN9NKIjA(A5PRSc^j$Rc~b6yEK*5e5JE1rKIdTb~Nw!FxC`3kJ+JNe`mq4=amz"
    "H#-lqoP;LWSX?ok+VK=PB=&~?GPuE;aLZwM_vo7dW>Pt3{jr04PCq-"
    "1b7|eLSiFQQ3HJp2l6Mn@=$0Mop?jw%mZ@Y^8e#L)rB>kkmupsFI9b8(GNu_p~qg91be=u%;N+?F^x4E=yB2g{Ck1Zt__QMmZKuG"
    "<)YOqIfC)#*E)jomZVSr$}UmdrgXHem})+eV9m#ShG*AX_moK@3HI4?&X3RKQ};sLM8gJwDeSR_^WHXQU&zjx!RB(e(969XYr;7K"
    "BQtJ5DVQ_S`wk;>`1ufULv8{99SZRq6}6&$j5Y6<<YlHJB{z)MyJ7|S-"
    "fPrGN5Tk(W3e5TkNi`Qlf`WSyqkY$Eeb}J`C(w85Ao2wu|<(XM|5^YUotQW|}rsu@Lw<o6RT4sx%1otCm%NALyv<TPtSS|ZjHrjr"
    "Zx_zAOj+0~Nm(;qSUqFRTira&mC`rx(D~XxhD)5h=qlc7RY{hpyec+KYNHOv)K0DztM$klW&k!)H6UQ_E#obq`!Zd?UtDI#%r9Fl"
    "%@{IKh>9Kq@7uHU0v)aCbo3x6cd7Y2*Ae6A(_S0UynG~RjQD(R^_pFF7#%w1-"
    "g5+x+K~HrCe>%Y;=M}CX$?Eo*B(62uAQWl>h3*x+-"
    "#p#+M7`_>5)P{zUAO3+Yw`dbDV!MR<S&}XGxh0Sr^(6@5}>(j?e2+$Tn8uLlc<Cu2=LyBI9uJ;ZN|&9L-"
    "{DB%JX@AnWpRl_+DGWTdTlf$M=Ri&F;S6LIqmWl^K*(Ehmf`W{IZqYIDw+An=%&LTPQwj2zz}6@D$kjF^LkCKx86nEGf<El-"
    "1!xpS*LZe5;$IRcnv@{BA|zxcKZK;TrDX!84hTaBWR4A5j#SI;s%@bi-jMk;I#U+{MJ;Fu}0)=?@mkt^m%wg$Uw^2=*(3^LO<g`M"
    "FxcPy%Qzc2y^EPo;hY_`^&V2}JpXV1O2<YLejXr_CaBN|FNvo$B&`-"
    "qJkt<=k*+D|{iN?}WSpEK;Rv*IoDvIj0)MS#8EQ#_P&ofvc0e|lE4lInVsx}_FN`?q#l07iPoF9s16P!a0SwUtY=%kK;FhAYE(>R"
    "ksYp+bF<-+Fg&=VRR{-"
    "hvW+>a#UwT?$$>wk|);;C;0?vG8^xc3hYA(A(Qo1~MtCDI?F3USJ>5cuSj7#`NIj@W)OM2B75(!)+9kdMl&+GqF%cUT3fj$8^_tJ"
    "IO|c_^z{qa|>nU2wJ&k;TE^JskcVA5zVniDlT!Au1*pclI*Iu-"
    "{w1!B%c1@neD@%dlMS(P6FAOUHnnB{oggxvT~h*3DQQF)Egf&O=u|QOqWCpG~{l}fKCt9bXTAR3llZN46wH3izL<NUV65I53{sv`"
    "a)jm42dVtk73f%c`a^(H91IcRw8eQ!JP%$IL>Es-"
    "#pt}AltI`hWoqQiAj2+Gwr(lOlNG9^e)B3Hp2;=4N;mOp{AShUtR&Bb)L=|&omp$`N<N^i%n3JD#xJOC`dxMW24?CAJ!vgs&FLOx"
    "I1=OM4Kf(;AiPhCtI>Rd%AnPTJBH&($K~s+<PfNCUpDA%uL5-"
    "@<#2D1h+vQZOEYUh9%i}o1T9U127n=yANgb{=&QLpCS}@h0`@rn_q|$kKSDn(`^LT^yN?)TRw#@fRkIz#v@gN-"
    "L&NbB<Qq*)L5e%fvsG=u65Y5&e|kBX#?&Fm}Q+|X4sC7QQ}Rc^i<utfX0~-DHRO-Vdz`MKa#NuY>Pu_KDo%>by4d~yF?y|l)7z5Z"
    "RLi{<%PWsc~1N78mytVf|3y?w^@XC&dBA+kc51*i%nTQ_6Rgm9U-"
    "uS*4i(9s}_wNtS)RbXh?r^Ml05^>zO>&y;&UDZ~L6~^&KE(jJH;49rvUji;R~ysR?W0${C+0`>i7u&upBOBCJ_$ZCEjkl<OMO#&g"
    "7`%0R5gc!v?YTN@Y<dh}{RKwKXA7nar2?{mtmxOw$vt^c_y<LXL2eCzIlt&0ed(5UU{Po{jq3k^WpnX@a|O+7|bldhMnvx5D461k"
    "z9Dq=Wx-$~H3XS>q!3C1pmA2IA<m$X)hMypI|nMjIA(4N*5NlW%h+{>40_7p}V_RG#hGwyg3;-Ij4du;yrch3x?-"
    "1VLQ;@i#+S!p+;)iVL&kOs%J07)uJ7^JahulIj12w(@#kwazul0*_OhdwdxOB+{K(WthDONWD;<O7<ZyCd1CTRi)g%)KX!)gPGK&"
    "}q5m@3wt4(OYJ&%e8H4{JV`brfc7C_9lBWNBT-g(vXZ9W*r}BUkOpFyS}<N?7-bfe*;(SLelKX&W2AK61-"
    "YVO!PMqff4<byfk0G7os5H1Cb*E6v}dpIZ~l4_6;*^oitBr6}h0!7(2G!nL_Hkp4ao}w8@K`P);nN+|Y$c+(@a6zB|B(eN1YE=ht"
    "Of!w<|ecXf)FM!sd1tYXZ5ZBRm%%y_g^Tx&b+0vgN*=N@T<)1F6NORx@WJbGlMC1LK;Nj+3v;Dg@UeHj^|&`E1*X5sM#DM^4rqGw"
    "tsFKeq%2karj$hDteG%N2p5XC5xaZ|Ul>K&uTFakP*?$C18KMJjt<i>*FKL5LHxu?>dtjB^&97o=B=y!iV($SD%Pw5jBy}P%GI!;"
    "j!jC54Q=n|YzHQ^=)-r8j#9~yg@rqO=CDtzz5-"
    "Ja5QGE~t~w*oYH;+wk$EdD0H=j?UwYdM%zlkLwlPSaW($Fp_KpTyk4pOQ*+jXp0?ucU5n_dXt^ARGx2p#ePypZDaQH6}#G+pktph"
    "QY#AWBR;LLiv%^$nQI+4(Z=MVRy{WgePNvcHM26gqXPwIL@Tk-"
    "ROksXqP>ai^$(Qj9_gooKHvMLZ0g*6koxLCT-$uTNC+^=7^nax~N}?0NqA&mMFgFqIrJLCCP1*rDh_W%peI9Bi-AxRk_!48+B7kq"
    "G=+xS9E-}pW1T`fWNH=+P_QUAiFow<TLBnoDfJLQM$*DMA2fV4z~2^XO5@iO{}C;maBL2rL#xP+`71ZLISyC^9l&LLsFeeCLDKL-"
    "P4y-"
    "jo*LlfMIF*&c%A&eddHpENWVpbB>?ObGapyNfPVF1`8B11NQBMy*T@bt8j|<)~v`nfdTVY$9d83c=r2lKkN5<GzQ|bSc<90x*qy{"
    "hsp)}{%aX%0{ephgaOAnsg{IqA&I%TTlJ?Zbtf>vRyty0Nj*N^{ega6*@QtCDsmdPZau$*GO$h>bD{<fnD~Q7dH34WS?W@^H#wDY"
    "TyD-`Y4)ydR(`Xx3vc8P*B??%bsCmsUXYuBEcU|ltWc9sg&EegNPqkoFk0#yr}x`%%}kX8vhf5qs4suFWF!2U9`Qh49l-"
    "@TL7T#~?@@do39la_%bu&Kh-FO<{QV6{IHXDc+_|(b6f~10-"
    "OrDGpgT^udvA=a*Q$27lk!mu>12!=bzhFm&H@KYDE^&;X^b=`E*bG)EOoIU^U-"
    "WKlYmm0$VyYiT=LrPj@Hx+n@%aG8?58)WV5#DcWd4gz=D3Z^O8*LK?w4DdfnfbcKj6&MH*yGdsdl|j+1W_m*F01U&Q8K4mS1v5tq"
    "<KFD={B1%It~j-+6nZ8}{tL=G9A-"
    "1}^K)S2mplXROX1ot@^A4UICNs!oN<u0&2Zc%vhRAj^^@o{M(TnxNUCQ8Rf19MTajjYOuYhHHxdYTIr`0Vwmn8+{)Tm~mX%4~H@L"
    "3|GJmHy_+$T7^0A>#X?eOYguy**-"
    "Fb3Z2Y!;)siP#23>Rs~%mX(})>L6MJ6z{++!GqSrvXS#LcBt3$fG}G!_b;2yXx7)unygJ}$T5gIuBfr|SBj*6{kOFo>b_Nwbs&M%"
    "yWY|!Y;+&c6?WhUkz9;g6iM$e-"
    "#}X1Mz_fi*=V*)FdCxHsr*sPp1)7*4l*f7}k8a~#?NF#hZm4wMcYvXQMJ8j!zxy`k4;tnTg(1wHvkk7^O_bDuH^b5X4pPH?#O<@P"
    "JGlsK9Yr|Z)0I1_G5($M@1$NuD@esXTL5zMEG_q?@A9Q=3Ov<z#oL6}!g;SDBfr|M+O-~kbu`Nm5QDLiisqs0{Za;QOGYjq-"
    "0N3>IR1mI1e<IzuUOwbA$kbQA9nwY0y>D=-o6=iy&+QD$+RHjtr@BM;P<0-fGcrSXD1v}#jpz;k-"
    "vIsT0fh&G1a^OXC2NIi^O&miQwGFA86>I*!z9G&j{*h3S$G9;zoojpI0n@XOk9U20ghY*I~w)uW>cLcO0R03(=VJpKLknYE<?7rw"
    "66p8S`o_gq>-xG%hZNCkeNhgHto8pallmJYG$C(4H$9?<{AhI>Ei8eCKN{Rg;oQHn9Hv7?KXkO<Iy{S7(5^)+8^{a5J|RB}pv-gS"
    "-y0a!mlKN#51w7p1+r?=a(E`&SHE&SilE^EUGNHRgHJdbzK*PaL750x5haGLJ}yTl!s_tKItmFo#IV3{6IYLZY5s5L4<Tyu?Fl{C"
    "Z8)=o?wIH|Y+SunR&TXV^0QR*l8uY2h0&AZ>fb5%>pp(K#|jD~|&f;9zp<cAoUk9Lq}BM6D@d7A;4hSBZnGBW=%an^GopiE0JyI~"
    "ZsQSDt}uSq*H#N09jU6L>`jgV@RCF|};;|E~b2-"
    "yzh0{R63+g4D36r+Ze97nG30Y!c%<J{Q!{?|@I#RsEvEe^!<NHCO*kO1w8Yw#exn_7pOPQP*j6JHJyxHc0@w(i7?&GIL&xt#|V#E"
    "{69>#(sOM1pl@K_j-"
    "k6*%`V#M)I=u$xkH+kH!WeaB@1(i%bc~<T}CW_p87}8isQm?&lZ#cI6Pdv0(?P`zvm4y<t?H`+$w(^1Tv4R^vHtb8(YVZJ-zR>{r"
    "vS>W@*VwHz4|7dqYEa_wty3o~Ebw@zZDDHaIoq#N_;8uVb3uP9$jh`kasQf<(+zpa1B9mZXGN0IRk2ZK&Z2Suy5Hs{CHWE5^t{VD"
    "e2dD62FA*+^)^a*EFdNnEdeYk2mJD$3s$aw`+ak4bKd7}UUR1R_bi&R|$k$YXtowIi*2<>+e#?9OvQn~^bj07gM3?Vc?f^<#5%&7"
    "R94BH&o*XNbi<j`*2{RBL$!E`3LD|0z0m;_m=EziBv#|TeT+XJk>qf8J`f|Ze&p02nJyFI88$(FV+qjSx*^86XuyK~a?Fl3{rEow"
    "!VNc!ux?A_k$X(21JqiDYcgSM}aLThV0&nAtRCB^*po_}|>xZFq%>RWb+B*IV)`F0GONS-JDpSOGqZFqB{$NWGlFlR`@$Dy=1i3Z"
    "hTq>;Oy$*yWBqX$TAV{Aj`Hk=~m8L%r2_t~(51&QGF^G-"
    "aJ#kvFIN`7R!9I?l)tXIyS#M0L+LgJktck8h>vP<_HRo&6@WkYf@mL(Z+Z8RwrW=s9L!M0ruUh36u+#T!lyA`X3^283)Bc11VXX{"
    "c?7d{J{C$q})>6=ykmcj&#?;4eK<PDsmw<EYL)N2B0e;*HR>D(LULn*G#pnp2jHzf*z8P8k4IQ~Z6K2pfq4WpD~mlOiu)0xK{Z@%"
    "loaPEDch6WY=e(b}I7}1>HS!Rq#aLrcva^2Wc^xTM{bPqhAcKZu%w})86>Ae~~PL4p9%GtIi=A>gl*DxlY@OH-xto215f03xn(`q"
    ";&DAJ@SWcks1rbaL<bf1iv9G?e|WHpkjzn)LmzbD75J?<JL3yCM3lRmnSOpyU|&rAV&8^K2gLuo4cE$&_{#r&+Ipc?BXvn}2+cD8"
    "ya%@qaE5QjZq6cjd0{9EtUp0}M6O{6ND<8yuTogHw0GdTW`WaO+^Jw6M&k7l-"
    "{P|(*<X!iBC$1{g={1`B|MTyrGQwGJhzhYPxGyQ`>d>j830eNs(Rsz$vy7Z(S9l_2_kQGpcq1F{9Vuzh3x#CzWI%r#F%7s|V52Jn"
    "goDx@n<5Xdx!1Rl-CF)2@ceH$cD(G)$!UE}Tm=-nD;e6b-4&w;XRKgQQfA<|hVo-"
    "p{q_V1S_1#+@0=56~Ih9N_Tc~`sd+ktA5l>Tu&qMFDfM{2Y&^*=kN7hi8lYmkE8rI-hU%B<n5w8BNH8vDi!@1rHEkCl_8v_HNk1v"
    "=0@$>I80a_x+wH<t*k1JDXn)o&f+v#rE#jlNaI*NFCCEVtmk2mKbG;pM3-"
    "nbUa#Ho%c_s3o1Q@yhlrRd1@XN*t$jXkvX^CoM}rRf=cp8>?9|J?L~%WMm_RQj3j-(Lv(o{T`%McBf<gW6K`ut_ne-"
    "J|}?pu>XkY#rF{vr>gZ$E`zLIueb3M~f7Dh=mXCu6lXhI`_{)Eq)0lCRu*9VRA6b3qU5@<(cu`vUFWrj@1Le>y!-y{ocPI1L6~-"
    "H=b7~i=f|R(G><tKRqNaK5=i#Fey!ZsOZ5`>#2AXT8;XDr3^0g=^Mh#mAt!gYjaT@p%<i4oDY|<<cb+gxjJ6&J;MGlX=@3-"
    "#K2EACl0s|;JA(CLc5)6)Ibk`vLkl_WTcH<l3ab{T+CPBn}$C}I=;z8vQW>K?bX<yJ(5-"
    "U*g6t#jxBM^jmPcJAv}}L%vWKE$u92ECv%j=w7fTw2K9i_&f#qa*&UECe@`r2yht;VUE?d=^S`Q>I99AXLyzBAv-"
    "R~1xwa7k>Ee4AyoK|;q^lblon;Nm6iK|ppy^Xfs5yDedp;d04x7}Hn1^$Mt32<1oRs?gMX)TIb7el!C;A=)ExVI2!@<GJ5xw>Vur"
    "$l<Bpbb#BuuFiJGT9hQleR*#T$MHkCB}@&EcO=Rx7@U@pB%xk(J?pi%PWma75Bv*ydoL;mImaum+~q)rr9J;{-"
    "rH@eX!Gkm(>P6^Z}bX>X2q%e?f6Ujhw#rxAXnViWbNz)rxrJ?+sq5wV)?c8SLLGpYLzJlySj@&DI^NnI5-"
    "{QKiGm^u^fVl(>O_3Er)ifsWG`}CXhVNXLxS)88%H3l>a#;zNmAIMxNlT(_4clyCtG%+^c7Rd+>#*3r-Q4^M_=E;^OIId?D-"
    "dr6tCAjW`WBXkxtTURU{jJybZ4$%5aai~=qKAVd&#C+Lzxumqcmt?&t|?OPG=HQaq>FvY+eXG7cSr%)dvz(9DH-"
    "BRd(@zp^W%a76U1~w7zQA~Y?1<o6GP8(MI$B|vu^&VFWJ;=X4bbStvX=lO^@_n;xBq&@`q7UyXo$b<|t(+SK>-8m~G}Ib-IR_j=3"
    "uieM`vUWtyVCl<uzN65g92$2O`V^Y1!pBCfU!YyUtp>!^`16_5KviJ7ga=MQaCpPgIE(u9I{yha#cM|j}FenyESJg_TYy7&Y;u<u"
    "WD?egXQ1xXjYBq7#VDM0raU&L8VjDd~=yD3S6{p7M^3A%&w58hz{E6qfc+A)bZVHR_|pYZ}L4M$Jfk99Czqcv?SrZ*cAssPz?b6A"
    "=`krXTeT($}73U-f_oJpy>O7#*=c?6mBXohX*7{JBtLR-#;FO#-"
    "!nO#4>6vP;lQP`p%$Jtg^U!krqzZKPca6lD>If_3n6WF1*bZ3Rjli0xUs50yq-yg9lLrmBPmjq{~Vtma)M@6_KaeB|%gwFLDqZ-i"
    "TrGR`U6UbC6?H<eIGfIPm8{Y|jvW{Q-"
    "m<fB?Fp9~;?4^{97T1ZejQRAQaP6$nl)a`6wI$$+i|X&pAykv!@UXNrf`4aDzl2@LmAW5Me%GL1GidVwO)+~sHM$+7oyZ`jkuc=I"
    "@R3%I-;#(<D`^kvoxx)6*+4b_=H(!z=?>M$^iwlgSJm^-"
    "c!p~cocX0i>Jyq;&t5&{n#2v{Y4%&hxFH*ZlebI*tp9gihCk(Ns9`;FiR*By8F9UNVHv8A&ExJS^ARc0^hM}$^drtOs=!GSLy@`?"
    "_L{cK=J0CF5@{FE(d80q@t-"
    "c6UwBJE_muE^?Dfe;SYoT}lR6?fZm%GSN=%Le%LA~7xzp`2<JzIpcRfx3t<{G)G97168>k<)DzX!|QVm+o-"
    "+VAsii1d_V~2!E=h_LWuKO)2?v1>2Yfw`1)?Hd8i{m`EH4@0in)|T;uOP_{&|Jt>(a2)uqojh(!YDW?6Z0>R`a{{HLagAa*`1fn*"
    "e@bI!g)ILsXd@EyQtqcTgpk=Mnz#%0Uc1YW>m~&X@fN`D;d8;YCf5_zl^K$p4=zax;3U@$fRb8fxIbN-"
    "T^LLN|iCl2X|o(Og7a0L0p&7y(K&P?8ze$Xc@?j0Gs9tgu%uChDzUedqI2*5uQ@wCU);SvXUc19ZWj6*YDj-Y?l(br%7S>(*|U_-"
    "rxBfs<4y#o^7i*b7KYIKm-4_yKS}@7SF{o)Tt2#+fvY2Ha+P`;$no+%kY2+7F<4W=7Mj-nmMZkQihq;HL+E;<|zJ{&Mu<!Sd$t|v"
    ")Sq8guuJ@-(5cQ(ToJuISFoTy&e83DE}QxXjOqx`&<!G_v&K_HJUhB5Lh!-%{<LvY#oocMONO!=m&79{!B+-w>GnSi|*GbOt-"
    "A+pNX`OX3%_{sPL1^QO4_uJOXELBJJM^MbVX?@$;FKCTxNYuV!bNsAh(li0p5{5fKN8SRZ55F;}E|jJHh)BmR7Y&P+t$F<`bmL5m"
    "EK%RR{ok{V5PZXr_dT(s)MD**1dJtT!51qktQy&ZbV6-"
    "fY(5AS2{H|&jI$ltBvnJ6#nV^*0!IcpdBzE4FK9(u$6Ip44;63eLfXvfGJ%<GAnFyhX)0Ipl=((UcdY&t3`7Kco4urxb<eI4J#<P"
    "vrWjpN`ItJAO3wnrg(H;Z5<032pspr<eBWF_6n#Unavhy{`t%7@+Olns1@7@97w7I(4;VV|>pX_Aj#XglbbtH4LPxi(g|(odjSTT"
    "_nE8@3GC&#mo6>K!$Xf^Gnn7NAnPnSXu<SIa0>#Ifu1-u$uq{|>&S6zmB~>XMB!Wp^JkO3bE(+*~qk#>jJ@d9{<lu-Qb7DIh&~()"
    "3i=12%{Az)HG{*_mSqb%MM&2G9ND@2`M?n^Ec{hT0k3!J`>bU2P%1$k;OfU3rSSz{t{3*&Unn)W%V?DO!>ZI2jVD!3gr+s#&s2dd"
    "(74BxfV4Sbmd7!X?r=57N5Z%L|iI=5Wf}lxfrs-xJGt?h^I0{vOix1EeI%&UresUetc)m5Io6HUBB2x0SlNfjVz;0z=;OhBR%Qu8"
    "rX66K!0PeDiW(D{iI#)3x0-9hmMh>qo*SFH?7@vH~R65#dAV_f>@3e^g6D@HxsT|CXT|c3uk{>0dZ-"
    "(v7#k_f0rc4k+E?ctv>A%#Or+^qk^X;wc>V;LCd4pAf|;j?CM?6Av0r;knej8cU1TZ)5zoQ0iiPwf3M1J%{2QUgjGBkF10SDg<}N"
    "&}~H!tXGD49!BdZUfQkgT0}&GnsiwAn5U1V^&}@c4Dt2fDphGuV&r;k2^j8P_;M}#&~%(t3~A;pkBwBPem9w3ke1TN)<q~S&aBBC"
    "^6t8vgvFRGm1H6|Tb8hg5-ft??xM>KnJO&_Sp(E7k7E?~%YXnKz+3XGlbKpYG$S<y14hWRH{m)uvK7HI+Ru*D-"
    "dzg}1KZmSZrNRT^l&trf0{R`GZZ726>;mi<$CYg3dR;U5#7A;mc{ZX#w1O3ax77Cjn(W7xL@H$g!pG*b-"
    "OOI?Z)zL&8g<{^u&K65eEuW6orQ&PeZ>9+c}82IFs~NgTiFGM4O!ynXXu=V@)OX5YJ?UTgI%}ZEj-)NFx}k<@-e%*l0_bvRbu~-"
    "UiEbC*~*-"
    "a$U$?!G~Vq@=e~V5z~{b9tBr*`0g>UnK|<Eg@n7h9d}3a4nC*cBh|vv$cfx%_?2WJ^}p+@o+Wk_)u~r=qP8E?DWh}N=4nZAU|Gct"
    "=0A_3N$jIl2<Pto(2n5XcLL959GckJ-!Hl)2IZ4=W@5h|EwP<&fwTi6T0NB=+((<VWrlQz8a2~gv<1@~y=U8!uW+=&_Bp<<SSO-q"
    "U#ozrO&Ds}6TX2kgRyf(SdqZwy}7DK2N;lyUtLZ7zM4!fg{=KS{?hpzOh=@lTT9Sk;|(em5X*Cq(Mh@vecd}tL=g2Wc_^m|T1}v|"
    "#Imv0G)f&2EnJaQGZ?u(pic$;?ug5DycHi<hzV1%jO()xWu?H9m`nW<1|{T<vF<<q8TpgBCp*wK61A0J7&j!p!+p&;cNbN~JHLKm"
    "dT&^^_`uuJa{MJBDBd0|-7M@$>LLIA3k`(v_U_p-^Slj&mm`{~KI*{9yUln`|7+sW$`qGMV$`p8)mdd$Pri?1k^Sq596_SpI}`n-"
    ")X-_OutH{+L0lKaZ0JgLduon&i#FMP?dO*6MtSY2=pgnrsg3c-"
    "W}<M6U~}hjWv+)dI4Z;L&SwbRxAonUIB*e0OSA^HScmOcs<?Iv^uVt&cS6AGEhnpP61&Wgiv_-_Np8p)a_)QUt=XPAQ5KyA@3K}!"
    "P&guqtp{JYkrCCqCdR!}*tv&ka9NACWzxuBGEGhbmbDTB{hkGfcbeF;u_hBhn4WF*GY8I|)wvKhr%{VN<HR#514x>=ojxX(BJ%2^"
    "`iG#aFI3xg{et&W6Wz>9h&|ih)GeiIK6E#ABLc+nJx{Vz$OKx&p`TH35MQ?B%f##Y2E;vejmOMDm>O(K@KDIF<aUX~ZQK()p@D^D"
    "jlF9doCa;z9#>uK(g+c?HKu?2<R@Ns>4fQqH(d$XRY(!EuW51SQJ{$2(rZV%8hRywYt4sxj6I44oRD!3fUAEX>Et*%=7IN@TN0iA"
    "miN4wT`?U{x%>{Nt2OmrV$Q~rN4;y$Fr7*+7>8%GcJH2*aH%IgGh;|zMXr|&7wb%v%n_C+TJK<j!{_pX0><|rv?e+7Hg48R)&H6b"
    "GQO$fpN890K_lkY{WuZ#Nte!ys!h0eqLmgL1U8R)AFSq(6g2;B+e0}d>*k-"
    "JAQ@fdo>c7!%AcVLA3>)OS@^}OK9!@9v=b)Tsfi3O@qL~>r^`EAcDYh+I}*O-"
    "!*Pd10l1=VXowVZiE2TB3=8Y%d+x1N5e0|r_fLh$FY!B@OB}gI*c1KK8IVd(z1^5=pe%3aGCP4(K8ly7N-"
    "({e{NL&4n>6txlR(A@y3R8hs>hM|tw`y+Rj#CEH=&5brzOL>p<J_DFYyjds_Xn!udFSgceqq~!FKG!yax^lUjNh<mp{SoH2xNs&b"
    "T+8xAcUVH)Hob%=qOdpnDvXEqK66ItFS&KJ-"
    "`5w4Yzg(1j1EKWD=lDr|i~i4k6s^my#8Zh>rgSZ?pwCYst6LsL{8XD1|VNv1PfC<1YBpLm~?maCWJ9Q$VV$CoMTXde?{C~u4LZ4g"
    "geB5soHJ`u$7yr{qun6iEWS_dzGDe(ulkkfaEwz~6DjUby0@|wW58uy(Rh><4{$InB3)}8e`i{N7>tkcT>u36(Ykx0M)k~`V-"
    "^_z`6u?KR5UsjG?B3XGwvvDVZr$42vIF96YjuDMy!m?Ud-"
    "|0=p{^}tei4oFV;7aF^HY54?f*&}Q(0>dRy7Cb}A0IiPWN#lfMiC=)#p=6Gfwm>?dX4_2kGEeBYVsND;wQKB%I<~sxjq}a{+B_~)"
    "^9ws7~9^Lds}Dw4|#YH#$KY@-"
    "llP|55%T%kBEPxUqdXvV)yTrx8D}T<IaY#Gzt*)s~O@F?(Vzyh~kR)jilPVzPB!QA6j)j?=V>76gXqG8JfVE?O&|P_m{wNqV*7?`"
    "W5#|zYERRHuO(bs}vS$mos7^N^C&K6OqXp;pC9Mj4shIgEa1#Gy`q_H@)HDzm7nta&cQ{)O;^~f{1${X-"
    "2&(_s_7%VBlyBpqzod7Uqug?w0+Sh!Cc#v;bW{X~}gDNyjV8ife%8Ut!UxAN7Z`lA?6$$g1#0?dS~l@5!6B!weT{Z<W9$X*HeV(c"
    "`iwN<te!LZZFN(hvqzi#a3jZjcQ;-~yvD99JAO=^jYbd<&uXYH|&-"
    "*AFOx*~xZISvK$}1sauv1ysHDsI$mmbn~#XH9ItZN%C*3O06%xLMt#Rr=^%oIy03+cMk?q(>p#}jt9G*LChPzSzSTl-"
    "5o$h!+uWWM}MbxNz2~2Y@5}rctfPzX@2kSw*eh~zswS0aZg-r-}V?!-_gPBiNw5(C#4dvW8b8G4t<8-"
    "q|f7D(L{i8N8<X97M6G|B_o{j$XjqvSpkK+c36q#MeP*p-ZJ*_RRiMqhrsG9Lh6<9hreZQoAVD-"
    "=Qa;}zSXY>T4M}FIs}~GcSWejpKBD>kCo{uDM6h4o%!cgHRpI_2X<ebA>1BclR)4RF>-G?sP=OrvjKgqO`jY@buNi_<-"
    ")L@efM>OUe{zWi0wWG`>MhvOH&8kU!KujV6LlP`LoFi$5TKa&u_ao^J$mS?430Rr_h0$_E%5^O)+c66NXuzh-"
    "*Zx)Z=)h*UXWua}OJ=Khc=#2VQfwC}WT2i=vFvKd7FU2_R2mgSdhHz3%TexH4Q1<!@G^vbM>ob)?DtDkf((8A-"
    "nObxh<=6@psbu5;EzXxcW3iLQD`U$G=l)PU^h;-$3Z7TXS^v-"
    "NJ6{j`+VbI()gK8NOFAw0Ww*T^0TVEp5Dzn7@r$m^H(@FmDe2mU+7Jc)OwkzKL)V?21sJNwJV&v5tnR_)BuREqwCS3RnF$Ur6C2D"
    "2#teN$xHeB&}=K5C0%J}|9k{)-"
    "F>p?r6WLUlHvp){s(gUs#3;k0E_W88Uu^1ChnqQPE8V{`EBG*^anOP^v;#9DSm#Zm8*k6I`GtAd3m_3Zsg+|?TNe4iTX%I&`fNj?"
    "FKOUtabv`0X22QV*r-q)7I(DRC<{oUtlJ43KMe`V<wVe0I;gZVN<UB3YCrGmC@6YF;-iL4SYPDxBQ4aHQjB-JJdVRy2$=gDfj-nS"
    "`;wzg(JS?!I)Mkbl5(t0GJuMLiY-D-"
    "*OtVSU42K*#hN;N)aAdkgikMTQ8`*KUNQ2;2I+{?628<CmZbZ^Mj1TVx1T`C*WI~)tB`#^V(-pXX~P1L~z@)-"
    "<Lq1&oGVdK={a%F!*iPZ5Pv1GX#e-~iwSzPwyF<a1~@A<_o99D&U=Fscl%xdI-zIQ%-"
    "Qeu%Mf)_wz>++3*5o5t6woKBcl|7fvlZU1SE_VcffUsWDjsWCyj3$pEx6fXvM7cP@+D@^vhB1@;u{E2Nx(w{4n7NK7sMhV~Xo2nt"
    "XtD>8-eitIQ-IcKq&A2M6QHObAZsKd>A21B{X3+MI4P#~c`gp7mm8h?fF~VHjlNtQHzNG}`x;BT_tGY`yxSKMBdy7I>rNx4%XC1M"
    "(u&(Peg^H6OptvVo9x<;4hI-$0{MnMQ?0OnWK#aYBfcn1lB<I@`E6!1y^Np_MkP6S(a&A|PCXjoo2X@3KhwIgTIAEmha+-"
    "bsl!4YC&KSw339qS(A5FdZ%J|gb^Pond*h;#l}_!+D>4bLf|qYgAmWv^hf~O)@u`Wb*z4Q@)-"
    "}RYXd^ew4uN>u;wc`7&l1a#`g;wtyQza-#+Vh$(|`@rZf~-^gea-6(Xs$IjV?cC{TXe`TCwLHtq%KScp42*1Lik@kI^Qo^7gR?-"
    "AVG$mtC!7_`S!fP@0g+t7#IX+7;QJV04|Q-54`H#64U*$II@hHFQU7EvvRsGK!Q!za{i!Lv9|$&M|;-X<l-"
    "JA4wn`S9T~yJf8p%*V#!gZizS1i*}e}QmL!mH^@~#I1j?jEdiTnBjHWUz1KF8DC@^()rx%}gLgp69?8T#@`kv-"
    "P5o0#IS~j)be~fe89Ba5V&s!~T05qy;64>Tc20@uCwMF^-imch&_uR2<nXfvO!&l|ekm9CM2{$*993j1O?)p^pF=zg_G#qzMA{-"
    "!hKOC`RMMm@7kH%l7K%zcEAkjS%Sv+U2#n<S8Eh+OGLIw|cqjKVtoYA)+CsWLycfV)4SXjhaG6S_A=jFM1^3#*D+$1V(q}dsE2_q"
    ";OazIdfNM9$`Xml$o}EEs7I4hnt#(q!)}mpMzZ|cph;U6r^9AC{t=Y`f>tv^l32u}j2F2Z;m0Msr^Mw6gOsNOF-"
    "e3H58i(h|n`PkLsRQw?#@@M(>Q-"
    "fXh3{}AIiO_`fO|_sQ<uR<Gs{n{#cry_O*R&=<X+)_1|xmVO51+UFb!g<O`H~&3Cj%3viLdF28(s0o%BAgRMGLE^>Nk@(<w~)-"
    "{+A^t}z)>5CtiNWiLHtv-jDe4FD+VCB5A89MHO^X$fTyg5bsp{0Ahrrh`V#YHoN8gQN*!2G&Wcr{tq;XP=qTmrI$n&G=TpO7{UVF"
    "-W4*3$&w-{C-"
    "bh%sMW89|RSF1Z~mnZA4~IH4>`4tl2)WO75}XKUK*QRai4<qlxDGE}+uzpT&$j##sF%_472|Lr$C(od#x!0qN}<_syv3iP-cUQ@1"
    "Gw^Lhf)j;GD_Le%$!a0V4$v?f8ziIpId=tF_UQ1HXe-"
    "T2IHa+6sZHhOLQNT$<>kXm)wQ`sv>GJdFK3IH4h$CS?F9NVr2Cf%uzsuok(oJw6tO5*!L6hl;}sm?5-"
    "sp?=tQv1yE`pSc*1%L#(w?$|}ms`l97HnhG<APsulCcamW8WPkX7)(PSPN{U|GFc?Hc@|jIv=Vmj>_-"
    "jyocfx+!)pecT<ECM=VGj3piE1Ehvrio_)kw$dWZY9!ek%&b?n#3I)wvKJ|UHlv_B_XI47vAf&tzd+GIXx_jvvN0zPHTK<A!c{p5"
    "uuUHraXIH_zE@*rj{&&b?8eteU%dNKcZGo+vQLiimDzfB-)iPD;y=)k=avdyLuV)g2k3gMb3`G=a>i3>_-"
    "a;)8juYXBRNzTP{OvO7VUeo*UA3d$%QhYd&xm5JR$ne^Vw%RPN@BX=O5o7wqQa*4iXS^iX;*=--"
    "I*cF@t$A!q6LO2s)(9nqMMzxn!^B;N1Rr2MDEFTaxGI^*$xLKHa*iGMxi0`5qED5Dx%ae?)(+_EpAUVc4h+=(9sIZJCiByjikP`9"
    "z!E15gl#q8r*>`k9*cLRkgm|>YWU88?zo(5R6FT3(jYm)v&2nmJgdYq>Jj^3UFWIfF12Pr9wjB!u<bW=26%tTx9LoKQJg_E(2E_="
    "Q<K3mTF7UP41=f#XCRx9Sqbtu>f!ALudF1?O@f(N}^As+%gyKrYQBO+rsw{_ezs3d1CM6bCY=w<P`wX=x?t0QY;;{7+sOOO5ueLx"
    "Ye$WWP^v)@B7;OH@t28v!H)`N~H&aQ6z+npR$B?C7m|Zmb7*DCN)udaIWHOCVqBF(2^h3y0AZ7q;f`8T86<p+paEQ<~}5CrLf}lH"
    "gN~=JO2<Sq@1OmCC@J;p`x98anDHmYEO_R+XU@3U+TbLFoK)>+oLuKsY`$Wzr*N?`KlHtG?!~d0iSJ46>}2T`raC$2t&v+p?<PY="
    "{USZB82QSnR`cK7GFB(({kY$)ajkvRJZswp&)P0uTaFVvVkUu+Q>$;P$v2+{2joESzf@JIq<*h`>H^$F&_(?w$E=)5WarkZwMWwE"
    "wAsu+8x1MjCY==cUlZt))WlMoM@|$S=*BZS)z*u<{~b_n`!@hbigm|_~B2&N4-"
    "Nkio4B6cMgBkGzY~fn$BZtVs$^848!Rpeu}1yuIve3q>fBhU1ZxDmDkG4CG)$nf8PnA{e))POl)Dm8(7vv>0CO2j3r}ESNe}kAt;"
    "D@P=DkBv<YSK8Y{kU=%vK`lm2(-nhA*V=`7#{_9H+A?BvS(wio7#d0Atx`?KHrcUfk2v|UX`h;909{oX-"
    "gKD?DRD+=<7Cg!O~2^}tRzkl2qZz3lmMQsQL79j1wlR<b**Jz%prqh--3Ey#(ov~+k<f2CkG6NUt+q%4dzqbD)%oX#fCbUixn;y!"
    "dWz?F#uT{RacbMU}uS1!17(-s_4oFn}?-AKT1J&{70A(rwQ<Iie?>@%3TZxbP9^TT!BkwK;Mq9sk-zn^Wesp0a4b#cVISGqkD0JE"
    "qGnhkg+yu@`kxMU4H*j}m-2Fz%=O$P>$&GzRJ|&0DGCC_e<;<!#Ua$^eFcjPa^Y9AvQZYols3$2wI(g5jVd^GibEL=-"
    "Wg=OTP1d+F3B?{XGoE|2MwI{O8QI`I-ZS5cAZ3Nlx_!JD&XzMAo~@x888o<H84`xH2R+LHN+Ps=JaRgiP%NU8ZzIWe*YMy-"
    "h+(%_fu@1y#Educ2AlK<Se6NM%jV}uE2R>i%hYFRTzBq4VUDwh<wMY<(Q&uzUa~84YZmJ&9XHsAJ9*vi?{`?m8?{=_%;n331PY~!"
    "sS#hV62TxyEt!EJ+~?$g`wB8D45R++HCaZ*_6^^#2CAN<R}59TtX9`y=sP~=r@`V!c3KikQUAU(>O3=9^7c+OzC`MDSBw4mNZIt("
    "48D_kRq0Z8QGklty`#Z2XZ+l{pM+u896@c%*H?tItMeWulRJ3r+e*f<CI7)xOncvd@o^?(bc;LbXm8_9+C;-"
    "0PEsFnmJu3&w@Z#>*q@o5@Ie*e%l30knC(q!J6>vK5Cs%*L?@ZjmNVK;ZgCrdqw{2Hx<|B-"
    "w#~1Lle7<R)Qv#FtY{ya4o!r?xbJ9|6rs)CG`wrT={@^^M$-"
    "U$DNX*`m&q%3WFNgfa~P`ApfRtA0k_U!;y+iI+qWLo<QXh0%XRW#YCZFM5(Z)H6EWh<lY%oq>*qY40DxO}3V6zp^Xag(^r|SsGfL"
    "sE*%-h1R|vp|N@zuCof#NOeJp2Ef$a~avb5^Y#|ij=(AGV-29d(!{<|tQXy^2RU-0ZSP*I}m=5I1-"
    ">JgWS7ZP=Rkfs7l^Ee_<WNdjxYlc^4b$L5W#<8cp+fA)QMQEaO;K?R&h)8|48M9g(ygKR1P~^X62)+W}Zi&a6KrMXaD|YiyfFEf^"
    "5Nvd1@4r#3uXn6H9PNKOU3P6*Mr7+T2Dv@6!7Z@5_xw&e;lyg3C<W@TB9GO(0*&<wX2~>{?!-"
    "{X5T>46=etzyluCUjJf$1JVln&c9bi`4m`J+0vpL<ig2t(<b;RU<lf|wy<7d2yJ!LWnRa^_?nNVmIR5rR57c<H4hqm4A{!FVX80h"
    "zjqp(L&HT%XCwo8Zr&Et8U<$H{pR`T_;ABnVGB5b=lB0}#dK`Yai8aJ4k2n-gK^MUl=$RN3>1PHeI#&92V-"
    "|f+XWF%C?XS?9p;~imT;y-w%mJuS}?4GxRcFCfvwWI#OOQP(b4;GJsx%<kVTpHVWJc)LF3-N__7iNpW&TF1f2$+-"
    "3WlPzhJL+U*L``;G+u`jQde#+)H_t7$%*O~wrqEU+)By)G#j`kP9oX&_T?~b-8F|0xl&5zHZNJ;_X-TOSbW+LCYwAbG#SXxVB>HG"
    "1_){7PU$)n}4!$D<00IiYx^MIUq-"
    "!!JDI{gmdiID>ZCPuUw3RTPxYE6xf7wVD;v)U(Uh{`M(|ZCL%)8sD@%vg$ie)?<pJ4$Q0eLO?dE<sdZxJ?2DzLQ=Aps@oPYu_*&L"
    "oNk-IGfeIu0&@Se}42$yt+~!%e@2E=|Thqb&B*<ZPlMebS2jSrVoD>9jB01~=%D?d@z&|9|`3GZS^a?~``5p{vP>^%<KG7e|d%f?"
    "krN;GH$Cyis%YH_?>egxHl3Q%Z<UXZMPJp?v&CRh_nEZoE?_*(ta1#rosr8_?6Q?NgC~mq5rfXt4GhJ;4zgjGwcS{k?2yC$8iTPi"
    "YItpWr!zp{CnJ-"
    "*c&T_mIg(5^HLU9B};}ziHk>)uEbzOqk+SS9*wqx4h^@nAnHSLO2<6)6gGz3)GX$VMC3lcJy15RJ@H=%A|fxM%Ufuv+FEx(iS?|K"
    "1jR-(woq|=P~(41W8Ncdc+}>r{zgDG9k(DV{T}O&4X$1-"
    "tMHCWndnoEW2XP>E_=YnI?<p5F4d6n*W+!69}Gt!kScycv0X@L1SQ^|FT<P$hRr(Oxd<)+|MVK+g3R($9NTpN1da0vZUV6(|V_Va"
    "$lX3-"
    "@dRQfp0C5)ead*FNMDDxUEAa4ldi(cx$G7UlS<rd3*YO*Bj>Y+k*LeueKMYoQZ7i4#=^l$D(o)TWmyF5p!!QIAjTl2rY27Xy7g~d"
    "$v1ksTw@v87{{*w!3Jnwi3wf6Jc@2xVk3cmHY0K25{1kDx;2-"
    "B?eMKh<i&L%~F@zDPW4#9e8A+?P82;VcM~Xzbf$Jo^Sw78iTSqn%ICQ#T`X5UJ6Rrx8OnSNP~2IoKE*P0JZHEM`XR&Z^{(2hNnXa"
    "t&3TAs4A8z!ox{OK=7H2)Vef8Jcijv=CyzDgTQDuHNd<YLJl{uLXX4Rgr9_j#JL_|ML*lW&&R1~e0+TWkPyjJ9S3mNEt;~BOZIRT"
    "nm2D(@tlZq>ov1l%nIWjqJiuDexMZ^$%yky=6US3aC2P0o#T!+;op5iSF53tqfJPdh1lEg${x&N=KEJx2Q?a%p1`W=9d5h?cAZ!P"
    "*WPjT;?!&)x#LpcOY&u2qMXFt)mzuI=HusN0i|7go|Tjkg)X^Ii<xhJmn}5)^%Dy3SG;()Q)!4dqQ=+6lpZxsj*IU<mDt^HEU^V%"
    "C}n*zzi;|&_l@k40Y{}Arql|ZX?T+fQ?$#mNEl6AWE=$(2mh-Y-"
    ">BYw&+C326Q}MC3wWZD8dj0?G`xKI9cy|@gr!xn6wmgJ4P4ZVXX7ZGxMK(~GEASVUH!ojk_{PB-"
    "#w{$Kr4)y6FIR?WLR__$Qjl)#d7)`4ANNaLRzb$x>f$5QJpQ(NuvBE$9dBW_lLtya3%4~+prEtDtH^_g|*U35^Q`IT+I9YElhqN-"
    "c{$#@yvq5KcZyX<;QFd#)eJ(9mgj4W9C&xIHR#nx6XI)Ny(iwcUel5pJI+6h{~;pIeP7^Rz+Ztk+s^vSnLc4QIyietE<a@CpYrP`"
    "De^XP)<(S37LDvjAkRGSnH}_wSTpI;8@u|v37D^DW0x<0w>_;P?hw%o#7mEFCi(|)Y)-"
    "g4%4J@7Imh=PS(7>>@LSQuZ?MzdkjyjpKt@HlbvOWPotEjs?FzOG>6-"
    "&zd3oN;a@2?KH09dZG2Bh^DL8BMFnJl@&RiIDKBRm8q7SA!ZppbkhZUOX6V~~pF?JrN$F_5B2}Y$Ot;#W#M{e}N>GYfRSrbw62@)"
    "(CQ{hwS46jm)h%{RRxlywn~wWFN)YD4=Nq#nYP6F5rh|@7<Ja)ry(n@Rr(4#QetGGbi#%bfrU_dS!(~gBXZ9NF<Qw|v!_hu|@-"
    "2N6)Et0f_kjt|J`3e?eydhthIh;z)LY#_knyMV7s~cW?h}bG#E3571B4oun6X-"
    "B425@#@PYQ~`o=hXLohH0@4<*k#E~SDaBoFcYOyOrtOU+H^V#Rz4;t=@y|*g(Hx3+H7#Dk?%??jutZHWP?-"
    "zDYgknGotn##(50cHj{9@q`dCMx950I5=Xu9Gec|z`FqJ2f<E@X5~2=Fo@FXTp3IW2Czzsc-"
    "~dMbjSsYAJ{G<7CN=Q+lGqD<h9#1PMybOj%4E3C2jr+YW!<oJp1d*m$;1oE5CfM8nSw?7GC*W<OtWxWz2<sVvjHOzZ|a`61!m2Vi"
    "PLSof;_tSZZaM;-Uj+mw~5vtHs83qe?sB<r1MzfWjWA|IFI!!h-"
    "cDwtU<BFd=R8Qx47_7K9`X*zO$x9$A0_9m1{7>P3!aYXVMWQ@g1iU8PZjAuXSa$fTS`RD~$4_l_fvUKBNRIbWqR5VP2eaPTpNu6m"
    "zc*bE^+?aeVNJ|N;%pm?)sb)DOibU=^%?1f6(XR8_wZY;YQj|BtEZvsu06+x>d7{7Og}CX#BP?8#_8ROHoGG=dRyYUEAWm28#>8a"
    "=TFFPglf3d0NY)xz!JdVbvGLX1A}ah3U|uQN{3XHl7sc`&6t79D}18flX3rMtGNZ`&?r}MHD*NmvbK*(>~W*5Irt}D;0^U>+iQhf"
    "FErPR%hHr{VCzIuZ)S3GXl4xiDg(quYeeo@YPkmxB$LSe(fr?}Kp`{WZ}#@rq}6wx`@F}!M>gDU$9=ZUcZPWu>e*(sPSE54dq2?~"
    "4GJ(&;(0Wh-oD6J0K$~S%u>I<p&L1cdP2@UN@vqj4y*V*-"
    "q+tX*v6Z5oA_IC;A~}uW4cOQQwZO!93}9|I$a*{U^~?Mf`NFxjvj)Ab!xHfJH4~Koz!Hc$=~s_J@2<+jr7T|4<qy$kFlmD;XB~N<"
    "2j-Fju_$0BCoBSM?ffhj<9~=V>-"
    "@?CSrqgpx9g<au*#I&3aKO8Z#vd?wi(=`|ok!p;fnB1UQox?LGdK2Ju~b@1q&)f$EYpY}Uw=^z92+38Kz(*4pCWXM(|(F5K}Qdm_"
    "}r=&}Zw5x^B|rk-cJz9gtMNO#-"
    "9HEBcjoNztS%`d?&wo})SNSopPnvgSRhjV(_`v44F%o+|a?drq@o0W6Pn*N;Onv4?I!~itkwjC!HT?w!o!Zk9_K*1!~Vy<S>c0Nb"
    "gEYYdW!wL7Ko-!!7mPr>6Fgl<qcH07*X$DkX0d@ECj@5iw!t48VTHGoMOOBQ~Kwi6%G_k~Ejl%&guu-"
    "fm*Sh_^WeNVhvyGeETcU)=n6IRDxd{gJ0MVLl?LJ@#JCl_{k$E@Uf{EUg_fpR^QSf)Jd*KOf$II&p(MR4XLE5f!m_YEA-vJ7@pM6"
    "8jQDEK!z%yeEXwaJ8_OT-"
    "%Ai*xrfd9>Sqve2_@4i6}^t?u2Zq~GBPu9*ihG!b}p&}^GxnEhc*SMM5SnIss62!37w#K06sj&;O=A?}X3w2%X8E3cGa0_rpe)hY"
    "3Ipm13!mZxWNqcbkn*3pWoW11=>RvjBLIdlqMnGgyMudhyP0x(=WXb0DtfLKmj%_0$>=XG24!C5&bEa3GtRjuNZUXslvV;@)_Qt5"
    "|c(3E0FEwUndc9iSEI2lx3-"
    "bOtj(|~*gHK^Jc5VYvo9yH{4U)s1Pmghs^xqRq_Qnmg=cnmdiR|gDD%Z>I?H(IRLLCeeNv^7ERR+hpl#JPR-"
    "fn-G8=jQ4<6G15T-o=SYu7U%5?UPJ8P7>g{9KX#oC)wcmJha_tnZy~*Vgy(V;r5Buak&jh8Z;8i-zAjI-"
    "wZ#ZCz#Pyzjo~E^QbCo|^t@G366HuT{o!oZmHlRXAnb93vxoW1Pw^WNWNs_%XWa^A3l56d=Jidyq?Bos%}RDx(A%!DGhp&sJ-qBU"
    "YT*ms+L6OMmF5?EG2Z2-NE*htHJ}ncDKKu^SlyhddA!k-"
    "@}Xow+%l=!n?8>(4sLSymJ6FgT$laRXf0bSNM7Dg_4mS!{eg(9;I00Qa!YI<*2)><LXBT%`v(Cc9QVW!Uf6!$#aYiQA9$Ja!#m@r"
    "P}}8oTHG=+rpofnU^JbA@Yo1w!&A6EQ7$FLh|o_UU~SNfITo7jh~bk#IKUzvFVWRRQ;>TVLYLn%IK4PnW!+efg3|n79V#^cEP|3d"
    "gD(?$6$K!p(qU!PQ;eYX_KUZG^4x&5VO}N!^y0<n7x*04P!COWoBRS(mvBhXn`PZ27pW%#U}R4vE37nXQSYi8{b{)specgicZ~{e"
    "Z72^HQBY69%SO0>_lslr?ZA;immd!pwx=thUkHmL{b8UsETAVJiE5F#ShHXFMcH6kUhF61gPR$UDpsS1P%KBKm%hG0WBt8xhBZ<s"
    "*sg!J4Ta{=@+!40b6sV?CZ*xm>rh0Zap|3Sxf9{(f&{xD@Ea?FyFJsYj@~w1kfPhHSW}7_$S<{$1HEda!_|cFFX<zu)-"
    "a#H&#<KEtzAyMkLCUb5bl$MSBA;$f=2(v@REU*XxAv90(3@r#Bws=H59{YLDXjc%#=bmLnBl~wB{s7a^=8QVFTt{dSr7=3t@Ytt0"
    "gC3f8G<iD!UMAY~df_<B;g3JEc%kS`7de@`G;HY)k9%~7Yk~91S+@27#iqOzLhl<GyemBTRJwnR!cWRR01nLX2;j@pwHEciGcQhD"
    "J8|b#R00bbQ$FMOYXkHTv#I&{ay4p{&#6cdLaUCc}Shbs}PTQfK>)~{ADq9)!8uq2LOL&^rzM}yQK;B%hNAo{e1=b9u6~JAJXV|F"
    "^+@tBJhwaQcx+P(}<rd3PF`s!Y-"
    "kxQG6yBkp==$!h6_^y7aW2VPIJ$&@orvD+eg<pwuE8qZp1Wgminb(to=LU3&YAT%N+MwfCRv8-8wHl1GTZIeJitObruI~(-"
    "713Z>w1DW>U?opIB5JQM5~$xl&cIW<>+~fnT4WDE`SL`7rHe`I|DmwZoEqB_m+fF^32jCaRj;$i_HeLlf`?YMdEhVZXg}5e-"
    "iW7Drsq-Bv#$Znx~<UTkK>h*zCZU5=E)S=#mRx*DGB;iF#k)Cm~5$CT&Eu-Ppc;46ck#(naV9_wGBlv_;5`UZUe2)Yk!hh&iOFsW"
    "JDKSDIYWe1EDT*RJ~L3fHFkv^5IC$0fBkDPEVKED*}8zh?Vyew-"
    "(iiu{rlwR$}HnOb%hkI3on^V>`W`x?0g#DXug*RKY3@no6{{Ruz3^{f1bB1H0Wcsari@Rk7ujZfzH(Fuvx&^98C0nFZucR@ch(_a"
    "kLH!8v};K_3mKIP+${@AWc$fPkmPa&dA2Z2{2*t_p$eruE8gwnHeH<V9je{Dt4VT7-"
    "@oURcD4ifYJf_t{GrcIdD!#(y(pMGaxtMDoPzm=X?_#4=N@1+6s&!G+(xk1A9)FN+^aIPT-T7j&})8l(rUcZZv-vx0On6tF?cZ5G"
    "P?=yr<thSWgdRkRP@isU!nAK-"
    "&txu1~WIIYOYbt0i>O%TP3g}>=*T@IxXWeSrCYK`L=h^O(q=6<CX+`nvIiPulA>Z}KEBPcqi#E<^^O$ICm~3nP7j0*_*uQ6hQk0X"
    "=E7x<?JJ5U9jii=$&!6fu37f9aMmxxP&|zqD&*jrP!y{A}VTYvoN0;J>G=JzBOZcjC*Ss6=(wtO^1}iZm^5NKBF8w6_X2M?|iAUG"
    "C;m)|HIpZ1Csd4XlI*w#X1YW=PlsWNEkAaUr*HOK4u%eK4_*nup_$nCaM36PnlFV59XVAG_&AmEd3D(45+ChEK^>t*Yo0!Np!!y3"
    "@iJHLL8dM~lEj)={;B)s<8!>Dw89Y9Nk<-mTVFk4F-M{XJe9|BMYRJ0j61C_`v@1Kdo;f^V2GJzk_ou|a5OJ`x)pvbyU-AHLxQ_%"
    "hHrtlCPvQuFbPt8|1Yc#&JllHVvPU0f!1!HG@8tX#Czz23>OBg?Esw?#{zdP<Pa>;GEdOl1?rhAc-"
    "n|{QQmH4(QMx3Gbu^W<)<`m#E!)V+j5&W5-"
    "0wOmWp23YYnY>qKD`0=pon*hgXS2gC#eJTJ*iX=e?W+ja0UDX9oX?cSAD&nxM{~v(dSp%|7N!$4c}+dl=X&_Hf>XAp5SOnR7(R~I"
    "9axMhsKCygyqT!%4@phH{dC*4dr~Nl5M&db6<?eK^*0RwS9R)KCVrv9Up0e>$GxQZ9$BxciKqpQ62Y2b0{A0Z>q*2ffIcMOYu7=r"
    "(4PJXgkAt*=wd0gYLDJ^H8YENc(8K0x>vdjV5PaQFxVPvWl5IVY+HSu!p;8hzdYs43nFEO8(W3>M~lCPt&qf`lnj~;V|zh;(^ZQS"
    "(|pjJS8BlbXh&@KBADe{bLLtP#mQnk)aXs?2X}U^qk3+95Z@M9a}b}(m>`t-"
    "zc1de$cC3EvBaJQQI9~Gy~D&$&g?TS5LJUqEs^Ezt0Q?B*|ct2L({y+DkTJ?6f6jA=RVRZdUGTM20%Ml!-R!t6wIF%^8Y{GU4-"
    "a++3DVb_H8=Jrb0=R*>=2i>M{wINI>IXSZ{>`RWRPYFr2YTH!Uxj2kH=4U}ywS>CjQ>4p+x{A5Se9(<DvMEM}2>&vF+;ExDp>1rc"
    "_ouB=Z_Ivr}CO0+j8Cr2SL{ZZ*uC1^Qz3nkiw=ggZ{3K*nsP&r1yR&?=prhXY`<wtLRC=X6v)SEfe!@anE|SN41hl(STrsZteU7$"
    "JWatEmd46~H8~W4}yyxDFY7!#WIx8s?=l**nzC;ScMSRbTm-"
    "j|m+ZBa*dpb$GpNg+sC)!+?Tc)R>1@xE}zhd8l1o*w|on~>??}Kc5KO?E#11%saJQ`eN-"
    "m&NROI&hKv59$@WMAZ!)*u;lae6AUf1lrf{Kr54@<0FOfBbtTCoRzmq93=?=J!7tKVFIbxW2pdh`!$mOJ>??Ztq1xx~^)zme5>@R"
    "mpD(Jx3S(CPTp3u}3zzV(Xm>aBqFUp4bE51dn~c_ODVXv<NeWy=Nw5kR}>Vwif9)erN_@F0M||3*cCQ>){xFX1#VLhACc+<DEb?S"
    "x+>=S3usqcR;RDsyCKBJ{l@A`dnaw?rRya^h_41rS1<o5>wr3I_koI?d(YWhdUdrUNKlia6i4y!RMPV;;9qo?|d#JL)s#!17$7Im"
    "`x~<zCQn*XZi?heMy1Jd)=qd10F`yC3g1XwhB31GG9CkYn?KNiS1y687+-"
    "Y{FFybr>#VobN+Ym>Oqfe{l@^s@S_|gE30Ku!j=&(4$zaWJI1hDEwp_Tz2Qi*vItz_x-Y!X!Y-"
    "AM@Y;K*XX~A0aOZe?5IN|B(%_<)6XJ-hhfqYn-Avdwg3j!T3@Mt>wYtSk1DL)0VN>~^uJzvyl;;Jr3rIkb7o_t@b1kP$hEz)Tm(c"
    "wr^OiUM`^nv{zt&U;>f1AZvkjPedgm`B%6y%w+>LMgSQ7zWCd?#j^5!bt*~VF*+yA!c&EZ1hzGg==?Kf7NK|QQ+b=6=5*BZgYIu;"
    "&#u&d-B;@|3h!aGmQCD39=QN`pd+loA5twuMYFV@@Ebw1x(lQN<R`!(q8@boj<r;GJSpIxH_CsCv+Vq33gB-"
    "R1$A2CyIGg~w!Vqwdj-KX77y_qx2py2Iq8(QM62Y8VCHY^EzmO9d*gRdozWW!_)|HuFR=l}f6|Nb47|6Y+_1S$&GrTmILPQTwwKA"
    "y(k`_EdoQc+y@3<#x-`41S7G+F4^br_!zw|Rx;I9RedyMmi7{c5<92|e8Tl3IqNm^4GW5*0t;3y$P?01_wnPvxy-ZtxmB^d!)nc8"
    "RMARHJi~-"
    "7<F)$QW1F_Cq4@DX<9UKGqPirzV4_b6uMf%DIo)Z@Tg{+`$y!4II<^zBisH!2njel`dVXrrdi8k2PR#SgYx>aolG4*~chiX(D%i$"
    "IkJzz$FoWg0dah9RXv*oN5QoyLSz&K%Cqr6z8BJ0Fnek>F(C`i?6cp*=ygw%^d!XN}k4ik1!~u;GRri5p`W~S`#9t*uwwL_v)Ag`"
    "bxC&O7)|DhE98C8T8DeHDWy3D~FL#@xhI8;x>K>sM}1XWR(8`5m2XU5?P<*_1+GX)+kv=2;I?lcY+0rg+lB~JZSW1*~VJ(E#ceMe"
    "=dQmCew6Jlix$hVoh$^bB(71Z`xUGB2gV3pLsri$?@+b(*0^eDv|W%@1%=JM@QDh2xxLc6rBqBf!kJD85R8%tku;(;au>(nfK}-"
    "H)%O1%}zvmXS?%-q+9#40>4Yc7PIHc+HUE+GR<a!#{8kXS>v);sFALy+y>My4LFaDaJtEpj38!v5)u1!--ec{wN1?E`9)ca%`*72"
    "WPE#ytU-6_s2h55w3h<$DJf!uO(XhI>`TiUGE7iQyquoB;Q-yrM0*&PlC6PK7G{L=MmnPpB|$Nrl*hL(`-"
    "mT1LUgo+`K?MmWQ*UAo_XxqVa6CA?^U9grBx8?8)1xdBzpaP?24c>qyp7YZ@J}3y5a%P@5X?oKro_}o7B9~^yy0l#;;>oQ&I;Do5"
    "1Oovf0NeMb@~idJn<S%c`@)8*`1E>B_@rIR+6{vgSRboZn@cw2K<zMq!d4T;RjeOYWG40^xnfARbMZ`vQ!n={dH+Kuy256a2Q?qA"
    "gB)+gk5L*U0t&tPYyAY%1z^e-x2)y1-PEGC-tq1sYl-"
    "${ITw)3Koj3eY(^ax@QPpcF$5Lm*2c2|4B~Go0X=TEW_;r^?vj_ph29nCq7rSy^XreuTDBV5B!XSbA0-"
    "(ZNvgklraBt`?~Xc3pCD?#Xqm55wx^jXKrmKe3>~xH7nmZo5y)Eje^Dig`I2R?QGVVPl$WEI4?hCwLc=>Ae!vRvjLx-tWxdlh_7R"
    "pImefkl&}@6FH;(Cwp`gh@QrWs^+QHU3Kshbq9DmH^ABgFWMkOFqit#h!Zo4Ex8$q=<1ECLISWuLfJ9Vx_7=U2I;qxXQ*~hyCfw4"
    "bAXvyN{DVXOPb9H{IRtv?tYWjh1kHShnKdF*_$7nk*heQvsgEBHDILizP}(S!A^;RzuN0d%aTh+gR{<2?=i8u1lE=!H_zpk{p^y8"
    "paf8Np!bjC^;ruevAM}f^^6?f(aM=c37&|4xJo-GxK8v)etnxjj)WSJ_R>GsALMp+PV07tUF>*S0}0vHyRGU3mD2eI4{(1nE34@|"
    "S0Rzte%He2IrAiqeDlIOp|oH3F~9mje&^4+#b)t7r5`dg!N)ZnT$j;RhS3m-"
    "g@N2XrU$>y1<$tBv9Y51v%b5_Fab*@<zomp)&H*ED8!v?fu6{E>deUoQBIX|V(odHG*x#IaZtvgy^G=^7f(-|v-"
    "XwC(OKND^=b>7(@;am544u!NGuNifHK1gyfrP>{41(_NBK;3rOllFqdVk6Z^w2{6NWioKuh&J*Kb0Wyy>%r$USG2p?%saNrXIf&9"
    "xYV+A*Yf2c<Xo?Z-"
    "M+mopPTDOhj&ZuzUSN*j%n<RcpZf`&R9LV1@%&rUce)14GA!wa^~byyNtIFUB2ONXIar(KfRy_M%|By>q6QcUoZT0ppZhFH8!)}`"
    "e{QWFv58Vta%Yp4!V!R%B8;GL{m$N6f-H7P+Lcf7z`!usOROZ<HHOCF7$Biqq95%1u4|7pOP_FW!}uXt_}PP@ATe7Y-"
    "0s+h4LtMSac*O&-0gxRAys0AgAjEh6-QrW`EyTcX2WNr-"
    "S@5FS37}M+{;=x*Zwr~(V6`g4w+;D7QL!pp%w`R!}m0fBP`^OycW>4!m3O6;3w$2|MD(J?(8XTa!hHV71$$$@!n^dhv36;%D=C!<"
    "^fLBAkCGMP9qc9Lh&^Bgcp5N(qvEvids|tN*GXoOfR4(RhL+N&-"
    "9#V2o+drLx$6?H^uB&bto+Pg|x6QV<voTdEY6`J=KNZitg}Esht|Nc_b}&LW3C=#UDPN5laLAdNiC=`SyT=y|wT#HtA0?^fYU#n@"
    "5s8MUjNW@-"
    "981`AFSXh4cdCjK<m`g&fZA%)gr!fIJvg{qPZEMDXT&kanP3L!#XZ6<#{M>IiJN78Ame$CuB!~zlpY?&e!EB>w>R|dPbi0xz^Xxj"
    "t0d@((zq%~wwm@yd}W@*)L@byAF_3HrV*k_`jK2C7`dM0B43@NsA6Tm41uce1_l++^r>Xo&er}Y&SirE6blFFWtj2X;fFP;pHD+c"
    "6qUa-d!&dY?u486s2x-(%sIJ8I04;Sym6ybjCKH69!b@%1p$rSAQbkLgi9uSEhDkdbc)}x{q;bE4W3cgjn>ETnrqKQ{1neQ-"
    "h8Mbw_U(JZRvt*C*lm;ys^P4)IDyho+(2FG`eBE&g6DOp18o5yhtSYy?}XQRXU!4LUSMjLfCE`3DFxP+ZMOxm98QGZiLJ`Q3F2r1"
    "REI3QS&WNqAE(U6c+^aAH`TZi|nASBAyIk_BnT>9!YPyPuxdV_cD6*{A0vn2^NxJUzX;q0koQ=5Yl|RlVw~0FWASuj6{_}?Hfh`-"
    "nus8g9)sh3o+r#jX?1XxoD}t-m#xcYP<@KR0sbXb=!mKZu7l;c3(pC#QYo$pZ)!ORwb_fz4mS9hnW<s9f#|YCIgW@LX8hbkNt-"
    "<W4N6)G|gJ-"
    "604yyf+W=~nSXTdSIwV~=tyCxX%63a5|Fv8k058H^$waF`Dlw<@Mzn*HIz@6Tjp(xvudR{=L)qE<G2+pG4g))6%W94+MDIf>ibyM"
    "ObGEvQaMp+iK85M3XX7m)3(R>^gH5ze^tl3U`YGW4o$_#*-sG@#K+_vLv1sA<7(Z?4PCY&AO$LOW3R!U*<#9svx{o`-"
    "fA@jolF|F?9<L9a*fvS1|9a_<gv>c;H8|dmMs9fHbDu?(JD(-"
    "8u{a!khbCftcgBnoLomrM9_hoo~^|(^FTMlQ%X;!r2Zy<B^gbp1AVh)9tW-eFjgQNZ|Vexs<E5^9vkbb6CBTOj%6*Ia7`#G4J+K&"
    "!LkZ<*Csq>Qw0u5Q(?XdHASG?4-"
    "v0YVjg+4t`66}b19pnZ(A2Vj=5cpu*I+2#g%<+oQjz=bU_l%IMDOiM1KZk_i#iLmKNuifbPGxv0)uGLyEKk#(|*aYPI93g$xv&0c"
    "<QGoF{iw9j?hL9~dy-3@4(hyjq5yN>4r2b{Thz)z6U-"
    "s^0Y(!TkQGTd2o7n*1X`ge%mi1hVI7b3;k5*kYR;)sDKyfw_Nm{Tk=Qd1h$>J~|;O2||kKpowK~g}v;2wGoImw>DvbXm`rkXWpMs"
    "I;@9P#WmqjA>|p~2&A@;0izYNv>2VHqs@6@W^{gl68X!_I(ACi(d`s~61d|Yi&*c8OxzReCy;W4#{6PXogscyLse5AANqQYpJ@$Q"
    "GgBfr)77~x+C{l9%)-T@-bj`9WYhnCzf#?Kc~%6h-"
    "}#;DxiulSKICb4GIF0@>%@edN4DA;)yI<l(BiDXAF4#*tw|AWO@{*c;Kfz#9Hk+mK2Q-"
    ";JB@WSM&UJODkkwW>pu)!3{qm&qi^BMgU&XK@;%lg6Lm6sGUeZEHWDv^dX_Cld^PYxL-"
    "6+iC1wd(ZG_wJW*$v2L?4Yu;4^?G&pXGuGw@ALhUwrn&QB4oDUzRt=@5g(jbdVY?G65Sqx_y8jq*f`N%A1`@Nh#3-"
    "oqbd;n#8AO@nmfMdU$^+B|(%k6GsLm=+(fHT-_%xKEIVp1%Is9-"
    "(UP#Fkwq9@y$pSXAF82NMf{LC{!}heLw473CYsZ05^g7hyt}R17hX1sK-MNz^<W(iW5Gr|<tlfRQ=vF%~Z^d{_D?+n7zy9uft;qZ"
    "#W}b$wgEN)-StKAHZ!6s-"
    "?~5c}3V%QP^C8n?Ye$%NL<+~dO8r@HZs{ch|%a_JnLyus8wL}*tdoZacrt>lfxi(?m;OIhzifBxJ`Z)JgSBKKnAi8wT@^mP``NS_"
    "IxiU?rVh>^}~9@4u%z3UJ_vv?!cr@p_$BRy$oY>}EAeJ5`r^t&17i5F#{Hwk;6CLVDz1hW1->lbDu0MdjAwfGYr-"
    "N~Z}xQ`oT|GvHq(M?yEDz1;9;$U<b@4Ga5C`8YMmbBMSCw)`P8P^VN^w$DQ8VO<LJDMRHwpzUiv+W>1a*30nG^m@Ipb8~gulQuk3"
    "VAHW%Hlp;Cp6qKd6|Rqiq;k#oq0(d)m@^yDwYhVWB9~`ZAl|(*$T;%z0AP!ZSj;h>c#=Ar{uIDHI08Jf{9Sb&S{GP@j3@Yph*Ju3"
    "4VTZmu+;OL{7?gYbh*r%u>!gOS&%X5BTLbqRwYh*q%@;3_`^i)VXWKb3f7L{QjnPI0hU0`%hBd8&q~Jg6V;ElS`(BHNY*2i%M>AO"
    "T>L=j|3NyqYdb#H8i&%9J#JpC>Po(KKEt6W=gP9o0yx_x^BrD#`$GL@LZDtb%llD+1bm?m@Ki=xYqDsAfR{J82euaq}iTG@}z7^U"
    "hXrg0JpN6lj`0(2Z*u!iU?NMaI_>sfk1t9Bpmk>UT`9?kzUXG$1h(ls9&47-"
    "h(;^dD!!wy5_3=#*j9!*WG$OOH`n2a|yh31l57?U?t$o2>ki7j(rQ4Xp`;Zc0HaTckKHViA$UNL>eDSb{}<y{eZDnTt@WAhhtxJ^"
    "cvWfnu95or94z`EUdOLn(f9%)d!JnXu$AFd3kJQ-"
    "wf4mH}cyXi)*O_iHE(VW^DCA&`@YsFcH1GywdjE$+>;|l3qjgJuR`A=~|j40So-hvJ&k$2T_zR+eiBD^4MG=hEPQ7Tn6T)rfoOtd"
    "Akw;FZP&!Kwl<}(neV8rvZ37>T{LafuhEazRVz2Kn(Cfuo121kM)|V#*XK>W?}aG_MMA%rZ%lLb;Z9p%w%w!KMQwFG=GoGK7eJ<t"
    "zXo1m7J+X*L+@Sty$e`5$iZU_j0ZbUdm1mUi?gjHK=N}wWKhiJ0+_DwuyvFNMw<!U8`=G`<TL^h*PI4^}*X~MH+BA>5R3SgQ#$`W"
    "L{eu!!H3#L2pU^`~75qwe&F!G*)hK=)H?Pwu*qdmPv=Qe0!@XGrVl0AV!vxdQZelb!=<N;PiMF`l(spU@{$M8PQfZW8xm`b?xT|U"
    "P(QBW>m5Vir$tGou<ys)Uda(cQ|y&42cz;xEwgH*bDBnb=gW`pOSZHnV`R&h$dQhd*;1dE(XgQLmWo2?-"
    "P2Be5ES4s_Oxc67h*);#Yj0tr|I!Bj@#w^JN>N0X@g>r!`G1X~eOmH1$gDmW^GX-"
    "=YyNn!oe;T2s^AHDB>D**vGRM?vK@Yo4%h;WL3WPX>40ykiib!Izl?4J|HX$UQ8iwN^xN7i2uwJywZ9m!~2W4kqN9X`&-"
    "MnGUTjy4#XBD;EfDrcTx0PhoPt`P|J8?q>T*HA%#R#-kR^0kzSINFnIqGWp-5;o`aMn3O~SNZnWxciskeosJ-"
    "y#6DN`nbzk_7pn69rv@uZMmgor`N?IJ>J*M#S}o{N`9YdIW_Hum2LEweVG*cyZiUJr%<eNKT8Fi*!G!(}NzQ_`%v^ww{wgHD^PGf"
    "H1uk{uDm<(Rj^HU<Li_Ar)}ge-PxpKNrFlHVLuBO2plnc0mL-uyTl1bnTHECD+9Ga#g$Y&#v@;<k|E^r9HGv`?sW(E3)}o05UV|@"
    "}Jbsq@kA&L==Bn>hZDPM{uOYX(VD$SddZ=5ry~|g?)Y#$4%GWQ);&(X5uWY*4ucBL?$DKX1m6K;gUEzO|d6Z8f)U+ij2+i>G*dv0"
    "OwVWa8*LTBYL{zs$czjG?X~0+3vbWJ>jbAjrej09vu!GhFf?5Sxg??d_3fnU_b9v%J#@$m5=o<|q8)p=|tyrpG4c2PPBU_AY6P`*"
    "hjSg9PL5EsWj4%TQ+Y;{@rB+tieWfm=%DKLnjyGYn>>(NB_SE%`6)1^Kn2IB2N6VApNAfdiW8zC+00ixFo1L$`w;%=V&fS-"
    "?J`f&K=hRQBfxNcdT5J8QLQu@$OjZ7dcpMrCtu$dyy|L*SC>OwP+VSVWhTiq+UOn4#=f6SVMF#!0nn(PR3-8K;qCE6xE-"
    "n5bptTZ|dSMQu(=SBV1NJ&>@E&JsX*OeiksGMZjtE$?IY{3KJSX%}=nji1zhgn+X&f<4w&zpO0|0B5G7+guUy5d~+AGGE=Np@6Pm"
    "az0xh%0LP8mv>Qy$Y1TuF}($FH@|YD5>m!mZ23L#2`k#uzYDtVAnGRb?o9jUq5^tR;~plZ{~wYkEfLd^JN0riqJ`JV3EIBkr`IuT"
    "t3Rue$rR`UI}*3B7w0&rThzdaZ*!T-"
    "XF)NKvUFBs5%LJXew=O3@c+AP1AEuW9TaaEHWRzkh;}@m+9c_cV^}oHFo`)`6RFjj4!E)QboX-8vd-g_ub~UaX3I4+%W^!G=|RVJ"
    "mA&(MufJKaZ^m&(3Dg+$bOJA(t@Y-ZF(710LjMP4a47<cbN&l}-O0SthrHn`NVZe<wl^o06Mqxq1rzq<V;G+R=HHLU7NTLWIh-cS"
    "U~0w|#JMoHIlGeNU_E_em6ehu^q_y$mw2dtpeH_PgR~5OkLJgdtOVpM*i{cSqUbCACy7RALZRl!#EcW0ygKlz;jR=rZ2lCJ%BA?q"
    "H<LXT&$357ht62yU!y7OjDwTp{7r5&3$dUz8|d)9!}75@+(1gy?qb`}?i7B<iv+o`buFLQ`^EN4&iwOrmH)Q*0hv5Nl$9e@7RX-"
    "9`Q(@ewq}8bVi%lcK&pBvW0>I6M6ZMdLWE_U7G?L3>_A%PiYO0%fD?Frg6YBZ%oM&qjiWwn?4z`<VWpNunJF<{hBQsa~G`6N-"
    "V|7_8*%WKZ`J>$MpggmDK`Sp_E^YFis;wMKx>$;G0(^hgQZRK5A8!X0Yc?VGfnm-"
    "W>dZo+SzRZfXkeDi&rsSpkJs)DB>OXh%4>a1c;hvJh=7|9j0VnmqY%{H}v#tZ;EpZp##?@<@TA<7PU=i&b87+M%}(iiIVwaIXoZQ"
    "`9EhA&uRFM?&FuO7OA?o)OZH!ch4z*X!aceeM8_N-o_H9kQq`z`UH$K*Y{fi)V1ze7ck5UfSnu-"
    "7WI?^iJ@8nNZJDYxGw=YS6+*tC`CPjX<b71nnBS_zY-"
    "&Aw?e&)|Rz2d@m2%0MTVMI2osL${&JgslBy<L$#yH7(|PH<hWTPltOX%PZMX$M$6}Qr5pY<&JV+B~R(d7{t$1M~^e~_t#tnLLM^I"
    "5n*OVx8!Jb<_Vxs*@v8QH)+HnzlA+^++U905&BlPvrh-?uq;4*lQm4DQxxJ+f#MqM#a6eDy^|#Xc`zoR--ArC7%Wzw4}nF4(!)qJ"
    "<X%4-"
    "f~)m<hZVr1s9BG_8s@&i*BA#F2F9<C{FRWwJkKqvbL6qqS8Dl0eq{9#cZ|!b*LeolE3wWSTTXRD8WoFqMZnXWZ;x>ZErw}tDfLv+"
    "zu!ACe$O!xD6rnYZ#v|@{rR>j28=#4*LH0|w>W--17W&5dvp(=gpPr|)v~26etuv@rC}$^EV;gI>G7dH<Tz_F>FJqiXw&HsgNgcm"
    "8xcdb$d*b(Ev`H)y3&*LAyxcQ29h{ht2OyGP_;(VWM)De8`C+|Y@!|MYi^Z&|1y!1zIK}iO@_1Rp*@P6SKH9@q*JBDV^0FwLOl12"
    "8-jD=X#vk2&p7}d;mX=v^fUY1bU)3O$d+q!<(j4KZM-"
    "F%O$3KIM007@BpP0X0e6xt8Y~>~EGxSCTImRk2Xx=cw$_RQ=$Mmiu^w;JZWw0X9)ny0!^on@`MqTn<t@Tj^4*EW1(fO@LWzFAY20"
    "J*`sew#n}C#CSo5}D|Lb3?#p55@;tzqk>p?%UX7si0>vgDyk2$jyp&PE@A*N`Y97(^~dw@e8dn741uHHTfec7hh0vyi<XPVe^i)9"
    "%W2G>wl-!y(#R$ZmU_pxbN-JC56$Ce<0Grs%koh321M?8ro;L)3sx&^fG)V4PhQ6V5Kh1~<o<h)Zr9KA$!6r3SaJQnoT|448$rEy"
    "qpZSuDJt;k<g#pNEVXmk%&S2LgTb0vi%6xkl79@V0;g4ilmsht)=#;@h-"
    "K3_LhVq8mFujA~&yI0v?mZ05F$h3y`ml%*D%O+aeY(J^@=j<@<Tq)t&^t=Nyfs)oX<Fe?kf<`sYU*RylN605T^l0PuI=Q=cx<NC9"
    "T(GvVRX1`bbS+I(X>}qzvrmgxsMEqzR~kK;4=t?>$nHlWsZy%u0i5JXo{}57N*|OJCk)XeQ_i%vd=+c5;I64&tku=G0a*`M!f?y<"
    "z%9+TGNxhV^jLr2Og@@GV1yap6e((H{OG^<RrlxwYwxeHi<Bj=_};<<uzio~5m;f*+MfVjogGhm`1cr=Ton;p^{R-"
    "R?apbbF)k_V7BYLHsFDgWN<#Kkt@ULN&>K*VDm7o6x%Xs1Cdk+^rk6Mys}Duxb_X;X?H|#7?8xf=`IN|(Ss7VVC+s%*uw>%CL8Mf"
    "fX1(WvH=i2jJX)aCZ@u$5@WY5ZBKlT?t+h|)`zI1k8f`hg&M=jpFhbB(s?W03eu6k5@OZjrpo8YW<$!B3*ON<%jZj?ISRMSBWOd@"
    "koEBa4;{Ru*?6vEO{Tectls+50#t#^WWFP<xS#$iG4GC=$Yw76})+3obFhOH%^pj^G3>a$%)$b4!ug-"
    "1brfJ346T_}M*$AYmQjxEn=~<6{gQ<?s1QZ3O*VUVXi0ohk?5tP~QvgACL4~a6jI4bFoJ<;xI`*=TP_^drZjU;n0eBxA#oMg3jM("
    "j%lXX#NFq=x!&LG{+?r;oP{iDF`o@Ek-"
    "Z!Qj8!}OK|=8F+<WwSb}3nWr28jY^BRA7RdV+OEroTcV}wmV)eno^>FbJ}@M^PQ|oAa6I2Avxd;Gy2vSz?Cp)n~ey6XJ+`u9PmWi"
    "-a|1P3-nh(MfVV0d{gFAz@b&&?Lp5|enyJep3ywBJV!iEIV@vPu)V%Qmz~}fBRPbG>L$)XrzKZKydO(xqw^{0&poo_-"
    "?cCdqoiZ=ty4w!TW=r_K1gYIU!z|}saALC!QELUXoY}bK;B{vAI}1l95By=xZ5e9Bo2(>a31?#G-"
    "uf4AS~~Z)f<GIz@>!%>%J+2dIDq@=|H!*N$@x>)|Z~T{GNOj&1RN#d?A^FQlcD(_Z%EOx3Az$Bf@q;-"
    "uW7jXqewoWZa3;#pg>VUByoX-Mq!Q)Lj>1)03^Lp;VKU9jvSnu@G#nLJOWenW4X*{_e9SY(uWL<DPvSOBFnW8QA$wTmY9H5z&*``"
    "`r5JbOf2jJyN$}=F1iZ7}^zmMfviEbapIl_%i^@aLh~IeyKI`UnbXc?Tqu2(&EryyKCC9V5cDC<qgqmt__`Uc0zkb!1Jup0N$oS%"
    "X{{ZV~L3>1eP$gA#=P6D3C|L-XDVKx+~x6PEuy;ChV<n{nB}_(<iJk`;%%{ALvMSyqMNs{rD$-"
    "Q^Rv7Dg2nd%bi)+p8rHGatv#4z8?3&)?pGPRZG@639noR`g3PkOiq+9w8wrPa;UOU6o$p7^sQO#(_7&qN*$i;b`o|PtUd$b{z+_k"
    "8w9s3pzdVrHFeb3hN#ayzSL60M-YdWd#lYxNB9yVugQ-lKQLxUfsodG6DCR1b;jiK3_MmY9T;PN+Pk7b6rt?q*x`0hO;b>KEe<!Q"
    "+TD~gBIo>i_P9L))izm`2I0}u8v-tus1e2Qoa;i`^I{GTQ!gv6|C=Y2q9GbwZCe|0AJT<s?~WQMp)Nj-"
    "`a5vv?yyYiP(f)Tud337-H0$r5L*F8Wg$O)O(LDGlNV^rLe!@P@_SW`^zW@l-"
    "$}T~bl!=J$t2yQ+&vduf0W~LZyHojHxU<)cXFSpq<XW5Yz2Y<N2^H)%W}Kw$qs;%AK)=PX1(`k*7d5@w}k=eV`#9Jy!_7x#moH#0"
    "?w#b<1zPiMg#9tgT_1A?o#mhz(?C>s-"
    "|dhqSmu}bEd&#3SFK=L4ygByfYq$#+B$MOnU*9I@Qg>?oP>K3|LRfnen%+L(@|WyFt%<BoFr)ZupU8R5GG93$w{Zk(2SNu-xY*GD"
    "uQ5Z$>t&B{DESrx`Oh7^LGKwKV*ZGIa8=upqc<)qiV`b}gn2Tzy1sEzG4@yaLXNt|>Ch)42mXwzOMcof?TtdKr%$#f&OJ$<gG-"
    "D`4uBD8JXM>02Q!2bOT1<78gq2TuO&kzwD_U3ik?wdU7cmZa};GK?j!ZWiApuK>{RM3+TV!sxeB$JH7|vLFK2GF}ELeVc1?KE`=f"
    "0F;<w?&Osdi1+v*)-"
    "VN4aM2Ch1^Ab865ZJL)hH;i(x6zD`W~YVy66Ty>T3fp*h7sF;_7Wt2qwUl)7%)V9(#cpwV>QGC~fQ18iCCgVCE$AD1^^oA}GT40s"
    "WfD81WcH*R2t77CPI>^)o5yLXqEduhr>*61LE79$Eo9t!dx%HIZl^=iUN~U>H<#_I7<*&>Oify;CjbPd;q$-ehmvJ9HfR-"
    "PAPxOUv83AybN*G<DMg*Ee)T2?PgWFfrLviCPINNugg&yA<9B@*!gIrly=6WT9;zZO;kiv-XNxNK>7P{6;o|Ai@GNiV*@-"
    "05QU!irHi*S`rp@?f1uZXV=voAGmgtD{;qXcZ5*x(V52Aq|o)^I9i53|AvepF2(<uoFz015EF>SZFFxGq$clDkV@J01WW70RGe;8"
    "z#Peq-^v2>_Pt{g@br68JuZDom}QE$PI_gcwq_wYb#%7YVGlbVEp^T}B*v#A)gn+Fn3{M?V=JPJ;awZ&hw=Qx^i?B_Op-"
    "w-l^X+IH9EX9s7GgcMzZHr-vcVX?}@LuVh=*5B~HvdYg<AJhC4XtFHg8M(Hw!h&a~aCXyOFq+!hU~SV~h#K9P{1y4nLJB@DZViA2"
    "I5Q|u8qLv5rK-=G~{)#+N2EXlUECCj6}NcC$_;a8M2-{Vo-DLa!eiOb4i_nPB`6X|Jy_KCS-!VIx-jjV?e*&)w$+>P!T80RUt`#r"
    "i4CGmR&KS%L(j!4dg%XAWAV}kR3$AnUfbMRhfwX-"
    "D`BnqOzd_N$pSCa%9c&JrXfQrIs5hMH&0O%Q^=iuzIby%kVUt?$3W5<zX*S{k4CmZHysxq@GtN3jJ!;1wB!!YcH0b@Y)kS$GnNQx"
    "jS3G0FWyX%~YsMdJ7fET-"
    "1>51%quU=JVM#R16dWf%#2`7MB9cX%|hMPY?v?!*bf<^PBXcOk}mdaS93{BVbo+)o#U_LWlp7(lqfqDWPXY5;%;#72MV;BmjOhI_"
    "ndMa9kk`WzW3FfRhC!amHAGAM=r?1bwzW~#(H-"
    "pzn)!xiH724~nv*;Xm^ezoKY@Mgmp1l2ua@){qu@_uVS1y_kE70f7l;gz;iP@dU3KVQA3%Gl6m4Q9y^pFvPY>2GQ_!ZY|4AQrR(5"
    "ymjEtDaxJ#-<qo|c3*gSQwA_yk<fG?l^U1ffDr<Nd-By&vjEt}VA@%BRw`VPMc5N(!ZXi713kP>ZGz&e<CiitFqYjuNSy1kOf6aW"
    "(cN#Bkz3-"
    "JM){4n#1ujYywm&EV%Pc9U^g<a`qO1B9KOc)ul64=}xucWWTXtQ5g>6;sSk6b&HSPOvW(0fQJnQie{Ok`GEDe5NCq^`6T~TRCw*k"
    "Z5QeW5!z(>#KUrl^0a!JqMzgUQelPTQR-"
    "q3GG3VXBDb&Fvf9|XRIfALdNcM(my@7cFPIX8<>)S`!dG5II;|L@6xquK#8Rha)S=0oZPWbaLi+RpV~OVK|F|at2c>d^sqq-"
    "Xd-i}nV+&gA6mGwqq#FnFt*S*t)i@|#b5z=!O>#PIyC-lF2&IXY-"
    "BZsq4Sm?&`ISqj2|_10LFyVXXLg$mc&&nX8AiEl>}%yG@+#{*A5DDxt7LO-jFdRO5N`8nMfikQgP*6{ggB#U}0D2B8384MgyGlPt"
    "d{Wt?+fDIb)vCv$+N|f;1!U{OwGQf-I3kuW!$TJyWHj_6Q(gS%-s(pzZ^udz3VQ8<>1>OP@2<Dj>*^I2ec83cg`&g^2dZ3~&-"
    "xuXXkMNKgYxm4_V9DKWK5rBACcJyk|kIMO0&!VD{#X3-olH;eowBN0mSs<KWfRo>vDq8SKIAp}{@lKjpYWj#D5rOi=?YS&iNNG-"
    "^sTR0!UMuyWNxDwR7nd)_ei>|eYQsz1a03$wU`9Wx{fn;G8oS}m@K6;@ij~Cz`sVKhnn2AEi1{-"
    "3tyWbw+A*>6AL8=t2DVlrk&Nx0=jGY8xoZgHaWBU&=&2Wod%1@-*1;!zFMm{t25H|-hB)jH}6hRC-"
    "6#ZD7zH#8(DRaP4LVKyRB`#+`rzoM0JL(vqG|TGsZ5|bBgZs+Lg6>eRUj~8E=76l8_!g@vmPHZ`K=fr3jlh{xR&%d}E_@fpJVCRY"
    "HQlDCEsAnA!YPMwYy-"
    "$vJAr%YtM#EKqyTwmI|{dpEpC^)i6$ZzKvZ}<+;jljR>&m;i~Eb9C^kJ!Cd?*?W$%?BNi+*l$)}2>=!BuxGEg=^KGD?YpYbRcKH5"
    "la%$iJ_mV~nqaHAZL(MQYDXSH30tDD_8PW1jS&Ra=%`L>V97DMh-"
    "(o4TWWd6U`Y^+A%9INsf@(Qy3M;ZgYF&6unxIuAdAt!Kn&rO3zuH&N?FA-"
    "{llb8Ym5(xfF@w^eEi(~`NU+Em|=8vPgp&E#dev>Il_nKj0iaDW{pay4w-z#RE^a%uqQgd#QBZoe!X-"
    "1G0)s3r~rviQBS{oZcpOcBZjRFnfFEMf=Kzt2TBS*O*`7rtNaS|YSx#`Zbcemk?oA<XGzePQ~1KF))zntu`bW=<9OfbX*#wt%7R|"
    "))`$gk+xFn5n)|22huTNIY@Fwf{{v#g^wsfxDcJx<Kln!Vmh2AfsFbvb*?#o6RxYpszFbW}7fWT?~fmo`Hbq$mg3;oD6khJFnBk3"
    "@^%Z%;w5>APgk7prPW@8?rvcqyhCt!LH}&cul=Uftc8tP9V+(?w3((mbbFsbf6b*{3aa*ZI3VnI<iTXKyW=DI;WT$xNJMYsJ;A1;"
    "`Mp^z|c^Z!`j(_^lOd#L#7+(+7EziOWXd!I*(#+j=FpNmJ?;4qFbUO@j5C1u_+a0(c=TiU-sFGZCyt0R;4rBV6#<kv6j(T>HiiLg"
    "W$Mp~4_BuOzoqKR5|aONGs%WT%vbW}AW^REwrmP#v$C^^SR%dYc2mJ?|3#N}wKI_bqP%-"
    "gk#m$TV#Uy`3!7P2C*O+U(6vM<A#;Xv@d=+4YpKk~T$%IBBzBK_Jw$I#EEi@ldU1Ah#TmnxO%ff>zL8<2%Kf5XY#3V@<@U4-"
    ";jV7!VJr5m+RxqsodwMTTvOIGD45PmI{`601h+w_ad@S;^SD^#tzF>!;r4WS@)z+CoP3QX0^(&zaG#?iE*I&?g9Wm@!T1gq2hfG$"
    "rpaQL_q6GQt=YFLcj(rVSisAS&IT6(Eay0lXVe;u0JXiW_(7jWhVDBkN$ivI$Qad*UgFQWV7SBV~V~K_ztn`fm_g1ZYuKFA^@j3I"
    "zYs38v3{W*#w}U+`BO;_lGrt*T;BB6X%SnQXDyPeYl1=a8FLDGr!qg!8=9$uV6n<_;An5rxVYu*73Y`3((X_)F9*4LVKci88WA_D"
    "97N<?t|*7}MGM5~g)H%3Z|+qHQi@C3h%t4X6ltuA;oerOOcX$hHHs<(MJTqwe)7xffPc@bq8uU<3RR&*b$X@Brq0k7JV$xVa4FZO"
    "=OHLr;NHeK!g_PI;K7_-rAFX28FQpWFiu!VN?z-"
    "6U1d0^7){oa!&uBYkg7omqFuZmDmBb0cGIJTJ4Qr;zm9h5HXTLuG?O3xhv^?P=^O&!dY(E1T0&MS}ruKsZa+Ai@rc#AUdOsjE3zp"
    "RA*BG#rv=<;>igNXNs7#2hT4cv@jRG8By7L|2)3o{?-"
    "YUbIBCN{KUOHX9o#K?mf@lLxWX*0S3&LvDDZAf^ns?r`hKE<z{h5OV<Dz{JecWd&!Tvn>??+)>>!X~`Lff;Uy$*F$jJ0pae_M9d+"
    "h&*9Fw+8Z;h1OR+-NmzLNs(Tp~k-"
    "cqF*|0+#S#{Hcrd%XPugo+wsW5;b)vIju>x>W5oXIcE<vQ9TuR{l+a&M154K0KdT_V%O95~Ud<+-"
    "KjK916v?PxYh<Rd^Lky1!b@L6ef$<2n-DSVdQ?DJ~jpC*B-"
    "!Gcs>v_{!NQ5eTK7&pqPp2{06*8+84bgU@TUnPMUSI}YF<z27QmdAZax?InrJ8O`fA5&H1X+S=M)@pFx=;eI7%2AlplHHdI?`S5f"
    "ijd46!nI{nt=kwk3M?Ct+<8QGVO6VQlRFG99rnCEz8tJR)BmGWTJ%x0Tx4rdbGov5?NN(QVk~F2ScTd^C>Y|bV&G!rPcvm}%Rml|"
    "(pX8vIwf%b^z4J?h@hLi6f+==m~XGOsaDF4@KRLmj;eXfb|k&yE|5RHb2K9ne>=3yq(^jSTE}s+XMh%Rip_>Cr}fNWLrAmhpAU=z"
    "Y%*35v-"
    "4VZuXsw0nQ@Um^BAy`v1<m~>}3~jY?v(DptO3Sx)h_I^zzQUZDEi`_0%Y}!^nNTL6TwHx!4@Yj)u%wV;S=zc54;}W?1ubAhPVP>R"
    ">TSP8azQ2X3>(TrFWfp*sRuXi13(#jg^V<`}70Mk;Cb9V`|G!dPe+&pU)#Bu7PC2@MX|ifysyn4@kxTBGR@%H%aq3fHs8**6n@Sz"
    "#vSRQNq|p0Q;74M(c$5V>UQSB(%I8_KJs6-`}CCRfc$yJ482sgS|ck|+QUg{U8-&E;!BuR&pS+`TWDVHBbaw(2Ju%y74HGKx_-QH"
    "%93m_fVg4O9iFn~zLPbvWaQ9vyMAlbRqVI$YS3dQHVN#1Ys;B36#cjlv>EhDJ<mT~6~1gvQ=o#69?ilDV~6hahhg*?GeA+`=!!uf"
    "Vy?7>nC1Yv@g)3-8?k=&Z?Me-P+{>@a+aG)Ol`MO>#lP=JmzwyeHIpQ)SNC;n_r{;sYxPH$6#XZjuq7lc=f-"
    "?U|bms!q&9ObXajW-YV0!{+bg&=Ot+7#hX+v>zmgonbr1Lwq^L^V1&s%-1spdeu(-ca`)$%r9KZ814-q<V~gJ>S-"
    "<Dy3!UCcS5}ogoYKs;}T4(URMH#(mQJ2m5gns<e(-"
    "SqDVRuszZj$+{2_CQJrRWd%FgBp8d}J!*IZ3Y<KMjTn$EFC+J%&;w4qg;LmMXOOMAABh9ku1Kb~w5zS`j&xF|{$3ao!7sx<rVSAY"
    "L_EP{K;y$TFnCs^A&T}%c=DmYG3v@|eO}ch0d!4%H>;n3X>PsBSLR68)j=bXj_QE&458PN-Jp#9KB-"
    "#={pxD}t%9fV*!Cg(xI!<7j5BdE)dbsEY?b~KaS$zgmN6C&QjTZJE(94E^j_wA2EJT|X_#8fz`B#O>(N98v-"
    "E6%O{?7Vuyk(Y1v|}HP6ALOq`VSZ+O8E~i-j2b|LTH-Fl)WPhC^Vo$s#pq9O5Gui;?0}n@P9>IHsW($I-dlcN$i;>CFc*!c-"
    "FdR4?NYZD%ey7p@)m$pL4%l>2`7ji{>VL_xe>bM!G?%$kNVa1e1sK}|^0K}==rX5ig8-"
    "?&)irC>Edd5wMwU`sA3KI{`A#5O7o2Uzva(*;*KV}cHfJI>m(t))%ogRrDSLMR>vx*~=osA;HK^8#@MItcQaaEucGstlQVKmOyIq"
    "&g5hluCsJ39UMIIrfeSZ3t%+3%zC1lE5Wqc&tD4FhYv{DitY<x${Q%o~i4UVr6V!&$LRu5c3gD!DQH(Q%mV-"
    "0r)#c$vr7<yRZ`uMn!|6(50u%vwZj4kUW^RgY+n=+h{Zx@xwP+%b#K9HQ?ij-DRPUATuIV3Py<|hM@s-"
    "q`6dUYX_LlSp+>vmH}&wa|qo9>Y8?23a0E@Bl<{r5VkK)Q^*KXOo+x+G^R}68ZYe-q3KB;4C~SdZubVH7ttZ-"
    "9hV8cLw3tCGv5?Cu;>|f^J;?~s!@!p1t46o@kp6N{gGB|s&PBv*n0X|b0>5fQ*O|n$;1p!**Jh(uw_+tt)!wwPEduON(uy<%yN?2"
    "5q>AcH!6=_q<yAUtSyLTp%RSDl;ylXgfg9}cwSW`O?@qEmF2t0C5N`mt8Y9W4m~CibRnY6LC!fyB2-"
    "Pi&~Z5I(AJZ(!qSnmEc(ajqcJhgwZaZjP&LDRu1Du?RH0^JJb=_7k0O}yI8J&zX{j>%{oawQKaS*vo_3;rNLQt+(=L$zq?MQLLZs"
    "(_RcQR|w=P>(3rF)=v39%hFO&)Fl2x#;bYT$4BWxkTukstmCaglweV}|cfs;MLuT<i|Ri@oYjWLT6pl@g0j2g*^_-"
    "2)d=kTmSL`=s3nSmM%$tc4KQB`A?1y$k`G?q%gT4LifB*y4128J15LsOUzKNFUm-DvM*+_6&3-"
    "ULi1PED3T=@04FP;;(v@bq3={iO`eio>xgq9xiC9^J7eRAdU7VLRLFWUT~`2|Qu1mCMZ{z+r&N6=aC$<`1A$Hf2olOz+K9VQtfPO"
    "D$K%xKKsKP*6(&{zF2!F@|OK;&$jk;`~38R=Y}sL_CYYH?gL$|A<;x(W;iu3bERvoPRNwyCz47Gw1<90IE<~7$sQQG!4W#d}lmsP"
    "#CR+$~T}LE?H+t9dUETdqs;0ym87IRW~5=L-ttuBeoZ6OqLmz{xZf6{kvv%-"
    "TpYNQxQ81JgTP=+;Jc?o@U1HQD9scj~Sy?VKdz>ohmCN#kDwcqPQ;Sbny%QL=!2ry14f9j+!y1KplSykmOiw&dZ@Vhd&#3IRz{j="
    "*rGvC_%sbG#NUB!B^L57C?GfVfbcHxe5G9?`QN_7QLPgLe#9TJooU4n3Gaz&ooCTAP#$$bT|#&gmfsG2c(W+4cdnMUO4x$XRE+)x"
    "h9-1eP-N9aB)D9W#^jY+Gf1`;5blX*GVUd-VK7%EXO3#qIvE&)x6ObWpd92T5orenPlEb;>dGRriss6p(tL8W97t3<Mf{MBuO<`R"
    ";nKy_(G!O(DipG+tGd7&^m+dLs5Gt!7(7e-"
    "axP|$J)!>1iNmMVwYNRJ~U`Af*Ot)Qd!d6Ci8vTfk$PZ8O?z+_qtR$Fc!kg$emjt)qD8cuD9JGKOSe14o@NvNZDlYA3PzPG!AX)>"
    "W1T*``kFTaa5+!Iy{IG+991V?;rC%R+KqCqxYuaZ^~@RpReq&UBwY|yc<4FyU(+TaiY;~yzV@Lhzul51%o*Tu~{ERv84<OBc(^>{"
    "ids!e`dZEr|=xt(L>@?3yq$$PH?)YPw^Ae+KoLO1qv9$0nBu?`*SPL#bRloS}uVklcA1>>>S*GG#|)&IW3R6_xK!yW>AqVqw*hC9"
    "zu#n7`ZK?8R+bto+G2%y@hiN`ZUP)3|c)+86-"
    "xd7exi^AZhZ=Nam5=oSj9NYxEN1LQ{hB2`+Tctii}$Y9_snqjb4+17LA7@;zn_BcoB#O{cm(X=XHn+J-"
    "U0c>pp?4z#)8RQhBI$I7YOUtjgP)ZN2^hdI(YY|+;+S4`3QwiD`BS&j3a4t>&c(VE!@I0BC)1?wmu&CY>jQugc6#ZkKnrHPM&Uwy"
    "mr;b8`(5qyeetB4eO(z~HpGBt}mj!uCe%R-"
    "!XHT?**;#pK?YJW_XHWukbj<qK9%y9wh63g9fKc<z1Yi7I264;&SLP0q=Q_94VpTi&>r406<4GkbniR;MIWHVfp^Oi)K_`85ryoS"
    "aB?B!3O!`OXfDmWI{EfiUf<}CxL@rg5RU@AfZcvLOfViZ0Y@1(5=qR(!-"
    "lY@J^A7nl_t8_D6G}g^&4l)f=R+IE$Ak_6yB)2vCI^?|amG69fo7PONChT2Sossbe*WnNqBxeaOaC*>)(=f4^USauRWT~Mlrp-"
    "|dICsyff>&c>rXqmm;nT3n#&%`YB%c10n<N-"
    "%i|6MBW9H`5Tj@JS?D34DG|uxZNY@SgyKbwB5?_ZXALM4sT@_Kx5TMzQBr>4@MU;L5JWE>ShyY5mAhx86T}1}Wbtuw!qC<&^rMVj"
    "iy58V|;ly;VYO7+4UUe{U^xaP>LUA&oM{`K5SaR61VoQ)J3llyzIdlr$R118n!2m)RB(d?J>ODttfuQa)CKdQoSq_u@SjwX_p}JT"
    "@$BvknLvJFj81i+gm&B2!SeZVk97WE%J^R*(D&XsFYK_k*P+%=#M>r82mcyH-"
    "2>PqTq@9?(H}6A?VR~_54<R*Yk?h2RnWtOStpqZd^)YgM!;^%Z5JX~~MI-bMdo>4(BDcWmQ>_mmAoqMDNMX)jH;{XGPU|=Yu@Sk!"
    "gxZdv$W593x@6}Xb{R&!IHs08WIG)9szVOki48qcpoVsvoGJ$;zwHtfm;RF0Ml)Il-kx;!UAHlYb5KK-"
    "=!Mn^O$@brC^i`ZQ?Z~q$`u-"
    "SsWEy?Lib9i#k0xo!hC&w5jOc8Q6V8y$_i`OtIVw=47i1)7HOLpCVEvVQap7+EAm<joQK9{dzsJ_Z)(^7E}o!F#SZtFRQjBjgxSS"
    "maQ0h+6BnI4vU*I<;TF3gNepB4@e<#$R}d*jL|PmH`%ktlTyLh@k@b^B#Xh`0RtFZ9UPi*}VQ??~+=NB!hz%KJ?Lc~KSqHRtl2;="
    "dwm1%5nJ+@tbymDJbr%fxK2zBx(cIC-DgnI!bR5i(@n|uQMUb%%c2Ne!XIVlz1t=l&e04_#sBcs<-"
    "LC49ontKTcxaBz6qsxVnhnP(3+Bnl-"
    "Z;Z5#42a8Xm`b;rzV0v>(JbX1`oD5jR|Shr}8=kT4dZXcM+o(Eui2y3muEJoeWFm%Av(;%shZDm?&A%YZ$3d0?iq7hM9w*iqS0E!"
    "SexZrb1x&rJG&OCFIC3;5b09^=wdkBVGYBSaqAEgU&|*HhX3r=|jJ<__u@ysD-"
    "4~1`vJ(@rPdLE~oO;QJo79O(S{<@DIkC5sWYh88X32&x4$ata^wL^lfQCh2AXb3Q0*U2;7jnTHh_jsSE5l5%k;8njSe#ki=%U8Gg"
    ">nil8@$w}XVEGoxIvcXI~898Zridk30~<SC6R#q8g&Mn-463%wd=hul>x!fO@>o8CY)aZ4>n(LsdU(nNBEOaRT-"
    "8RyWWDo&o|LnGk1rz(ny6JsW&97vGNC}6rye;kEc9Z66g9y0Y!DUWxF?eD}B2;ye1-"
    "_sDd5<yPu`Ut%cVPx1`F_SK77h6SCqXcaGdQWUjEJqiKS1WHe{Y{Mmn$5Q=Xfu&HeOa6Zcp1{-"
    "w0QEk8gSV4aKSM!xt7s>$P+`&SvYqJ4hGVw0Te0Zu8nP<TWd{^?h@fm95--5R5Xd1yo54R3kNFwaA4lG5uXXqrK2NE9{bXg$UpvV"
    "hu$;KblsXBj30*B*O3&!$@HG~90Qp&Spdc#(gv?5uX^ApJew}DDPk!5(Ddhql0(cX*#zIHyj+uoug8UCy0S(#Z&gZeWb{dBuvB~1"
    "iRmGDajTk&HAHBV)8-Zf@eN`ZH9rH)BL>Dagz>WCLji1&w_1<R3VCm;q-*0l#W`|J-+E@9*vPsz&MDL-"
    "uG_L0z`ZkQiveBd*dbX8S$hhUF^F&GvvTA}Zwmgu)r>FO-"
    "iadML$dTr^r%b;1o;!l3v#P8%Mcr&ANvg3V0K_Q2y@;f&*#XL?-lJFm9#p8kg#wbR<*4EEmjBmcib-kw6lQXotXE-"
    "^+<6fWQ&!V8Hd$eWa(I#wF4ew%hvMnY{+P1CkifZE3nFRTe2z-"
    "<^?hbwvI*A?Wm@V*+WpeE0haO4<!Y~{;2E}Hwfkk0NCWkW)Y4EdZWaEdXXY6q%Ih1k&Plt&yRV~SnEz{b2P&WIW;gG^lb8*FdUF!"
    "8fhLSZNpM^$`!=`mIHHm@L~PJArKNAhG|_i8`m%~=<2~6&o9O7R@y7_E0_<|{8YS^VNNgBp%|OdM?53@yFy>}Tq|81)pn&f-"
    "C@J%rfn&VLv#f5g40LbtgC8#s1K_L>@<D?24puQLgL&i-U}><){mjJC??cY^qoQ$TU+9E*<irgYU+q-"
    "4YOA>k10LR%1!M`i+5E}xq43n&(Ts$y`u^DAZLZzn*F|#Ur200#=#7<f_a$9z*j{RNaRq-YoK+m=JMg2NIe<_A{7L)@Q`r~I}Lq_"
    ")Stp)d0^&SBpPMOsV*DF0c975?ix~c=1ZKI;+PUvHTuP*vGHKXJlX_K!3@NAZ6!nAJQ8%JCVRMiNU>yuP6MW=roejpK7OvLoF|7K"
    "N+LqC`o{24ESiT2#%X7Hgzz}hz+gZay95Y5C=_Js-Gd!n``p(n1#2{d?V4R*qr;3`#2R8olBk<MC!Qxt9=X3AfYg0(Vb5d5YROiO"
    "t}W?B6$zar1kPNXx8Q~ct%h#L!0}B>pk`d=j069ncux5s2|Tp6P}F$8XKW?@OoO4f6g@UulI(}5wNq4fnh)q^z;c_-vlYh5^jN!W"
    "m0&^?hoB51?X$j?O>L?j9JX?QkuJ=mqroH>ZzT@cb|LsL>4(P#F91a^bml9xCfTGi#p9h5CP&AUp=gPp9fj(OW3ChBIpSLI6KZE?"
    "Y0gbK0Hf0O@B}erKaDsVwu7@K8d&tP&JtkW$4h^?+i*pRkfda0;><@9mz}cPYm@KvIgY?RCmxC5B5<k5lw)1Q2n4;A5%rSiX~&r>"
    "Kak1(mZlsAZ`WgCw?27~>_U20Fqh|uEGkx^U7-"
    "P)rrE65lFHUzk&K8@Y=^;tuLu&+N#vhhw^gPA5mSczi7PwPZ%kWvcVNw)8NY=G1{f>zj${Es1ZAsZGjBP?W&gCjx8uJ~P!fU4#&~"
    "h1w@j{8a?KbOB;=Wt5gm90C|$?fHBKz}-"
    "#{%gUP*f|k3W7CGAL+Ary|W%jxz(`DrOF;zwm&X7toNw>5J+eQGNxzuS&DLzByW@F?*d7c^pjePL|rRgV$_+ZKb!ys>CVZ)cY8uf"
    "|Q^=MdQ7gKL7v$DPvpgJ!w@!&=j>ZPN=NJVOju#AUb|&!DLY$B%V+F8u_(uE>x_Uc2DA5WMYs%WR&g$uL`b1Yn+@TA@@TA0J)lOc"
    "AWjFe(ib@-"
    "BFV^%MkE}ga?h>e@<&cs(T?D@hwub%jU%{SS(lM4d~neBuvIFWeXgU%?fEX>N|uo+&nx@JY$77IXz&pqK)Kd&={vhh^cj1D<OJ!F"
    "Qs!eYQjL;kb!ZS(sD)H^!}*Mp+%EZhhvqNh9-E_*6S6bl%#*pCC#=2nzJ3s6)|5u=j_-Jn&uOh$!r^XU%)LwQ&LsZs(*uC^6e`Ht"
    "%MtjMr#`T8l*QC^5uwd$!M%JX2j@nPmUSUiC~O&8UaijJ=+UtrHMXAF4<zP!wVaS2;_+4ECV*8CiYlnDKUA#l7$+K-4?=1cF9&x-"
    "3?M@+D5LeV4!Q=X|1BFtsyxGBM++QQEF5koJF{;G_@FzZJ2v;CV5O9wbjz?Ktu1do=Yz$sPojNJTl9dIXZM9G{%*IK|!L@4R!3<D"
    "`^Y>1OJWD>slCc^b^RWMuvz=O{J-81XOy4v6PXTwa6iAC5-4m#M~=$ZlL!Jx-r$n?XM)(m8svj10m71@-"
    ">K2==~*pfO7>~>w)?)&ay0HqPbK>^(v{EDRwET=iMcm5(#a5m5i)`2mrK>CSu-"
    "#v*|O@@Ws&vjP1&y14d#pt#Vp~@sVVW)70Yg2$(CKFEc0bod-"
    "M#lO5mL^{8_vXZw=b3&<kCd&gFsM{MH^BLI*l=n;*RQ2|hxqrEMp;xOsSfo}3Swg}EpcevJvL|uM;tC*aamxX2$*FMED8)4OW#OX"
    "3m-Pa7hLXpO48OSRv`?K;%q)NN*jri1*T%g8CRXFdiH5w<9&d{%nN>1BFG=UP=^-vRIy11tP(*De*HOXU~Q;9}dDq~>-"
    "UtunH4Twp%n)0Vl``3jZU8IXpM2tMaasM)!Bl-C_%Jc%)PLu-x(`5HSLtO-lF^b7;6V=6HH0aE5Aagd3)zn$0AE_5NknB)bL`YOB"
    ">a>6WP=(?i`w>TF#Puu*86yPImyswVvt9)rI`t5zpcg(z<!Nos=Seplc@V+}hqZ!zE#gjS%?(a0O{>se(k@_<rv%L!LU(1wAp<yi"
    ">A3JHG*L;AYNPn(Iam}Z455sbiA2u;$Xa0o2`*DOf2tTv)$D}wi2D^iUqrbr(m@g39tPd^-"
    "0=uJpao>Q`z*w?TINkMPmMG^EiWTdq|`^LHNQ@=;1&uyM%ICTgnfJQ24c+ss%7?F5rNPY(k`Jb)NWw1@^|$`Qw(;dYZ{)$T<f-"
    "O!?_8&2<P6JL?4C*WON^*yYlH~M1be~gnM7=HEx~5_#pbcIr^dtjGLR&O3Y%J%YutRFni`KRXHAA>X>At(hR&E3;y1+?$O)BZHkk"
    "5=2yr-$lwb$tE99T-"
    "p;mY8zYaie2iVjdP8lT?R`=CyVS@c{%$G;p#hrbo1wI5KyVrz1|a4c`TAog)r=P96vWTyf#_sNwF5*UZkwj$srEl<tz7Ay5;)@*&"
    "A2o?i>GE!TnX?jR=*K}v*B08HVox@YvH?Dqu0b-"
    "7qiu<`3J2F(I`1^!*oQJ5xl^Wf`5i!NKd=zqeo~jSaG3wu$eA*W(Ae7U^80eS(&Y|6456#4>&{T2|U<OCcU(<b5X&&O*|Rut=Ov;"
    "`6dVB>+8PkBX~e7C@C6g5b1kP=vVtxS0MpDg+YI^@i^CWRwMapR(i>fvv`-"
    "*U4<d*@O5Nj(_jRX^Y|uPw}qA}4hgWPSJ_Bt7gBd@kMyMKk}dJF!qc~M1dqKXY<ky*hYV2UB#paJx&vCXrl#wisAsL&)9fo?rA0r"
    "Pa?b@^9+g?6?0OZTx~D-21ThXFh~P^KWT-"
    "(9nqOyWeBPBng{6BnXPr6OtANjvWh&X8#%k8yP_>;ztGUho@UUCQ_K&PI$+9acFid{j3={Xq5z3qB6rP#^nfyUT`aE3cjXvjqjsP"
    "B>XZn3ma1ax`iiDXJB1<iD2)yNR8ovrA0KEiSN0DCvZDkiXyG4dWNQH5FY{p>_OB72r%r#ghW__gJZ?n<^$$;6hI?)&i7xdQbC#@"
    "oiIsEcQIVFZ#ctiWC_8;I|s6kmXWQC$diZ}~natNHEMY4=QisRfvY0oTg%W8+NZd<eek^}>&MTb*(CVtV`o+0C7p*(vP44b4qj3)"
    "@gkAh+v5w5{5xqgjN`|07iWbd5X41MO9v+tEeu>)aKY}Q`7?+7sv&Wh2@)*7oaX+DVBlKIR&OWBBG1nqO0D(K=~*MB#JDcebgQe&"
    "jzbk~BXF`Bl9X0A^%dHUjL&6XFk@$|hTQHL~(-eb=JAfsq%H>jBRfeW`E5-%R6EnT>s);d$bh{j%1lFMMvCT`8Pnjjc(T-"
    "jRXpoj_bI>{j88A~lK2u~;&u@=XA!^9b5OoMQrhq}ccal?P8_cl&O)38q-8?=)7m!oY?a>R0;IS*$!@h)b5-"
    "m`0hkFN*plyxM_3e+WdOo)O`;jf`>DrPudEo(usIp{;v1Kt(WfoT*V6B*8YXVFd%!${D>&#f>)4*gB5!A`MePFfeCz1eT-"
    "<+jn4jACeQMM}P|5!*CGqghFPJHZxokuaBl84qa?*{Ar$PrFT}Fy@OyiF0I*mL(xZo5)cNsoA07!PKoiW284%efxCujjB>+U;r3{"
    "Bjn9HH&q!)Q$xp)l_b6WxkOLQG5cPVFy~a@==L;-"
    "Dy`Wp)yFt)p(nwXC#EDMQW3fZNS4_Lt4VRZB&V#1=$KR%O^`XS76#JJGKK?P4z2wa$!^P>aWq$`vDl6|GDM0V?qWR8*Pa`3u&7DN"
    "xJPD1jhItG8jskS?7zicfW_IgdGd{OeMTR_?yV0dC$T6jfw*%TX4>I`h^+}z`$p3^`EZn*7!?GPMJIIy*?`W4@9U^`+7g%yTt<L!"
    "D{6+QJX%#JxQ{)JABQ&H!!X1g@Y5BfMYyqRC8tc+ILHa)=SO59!-6Iyjm8W<yBuq1soxCVDJ2+F7v8!}@Hxt5K{L-"
    "Z9QUZAC*YNIH$u>s?d#33WbmffCOYp^TT^;T=2!X^)jD)gsoA-"
    "W;^{h?ZD1wgQItG|C}9r8FZx1xE!{NfX#*>LZkxwfAUial^N7P>Ko3pp2@^2Jf_)vdd(SYGNo?y*Au@z%M^9;R5HoRtzW!F7dxj{"
    "dp_Ev;Dg&OmFgPzr&~YbBBXV~-QfC1ayt^mi(bN{{j#^F3=GaF{>5Qo&k0_(xA$K>A9$o6XX=>;S!x4&RiS!)fqAQ}4H0kS9F;(-"
    "Y2qMzfkybGFOSJ^&U&$4apYlUaK>6rNQB?ITzxX1VnQ#O>62^{-j7_Ei*-"
    "Qw>8k{N2Oy1f#PqJ;t>DcCOZYZaoS1^HFF*KM0gT%rc5LuJLrlwwBQZceQGQnxE1kst<Zn}!VXU=-"
    "ZEg?QVpqvVkdxd6U;GsM;Ew|kDo3uRwqNpO)KmF;|!=E2s|F6(C3k4L!w(P}zUBd{rbCgYm4L$A{s9Hd#mIh{$jftZ&IQVvm_&w}"
    "G<p7;-=TvyTRZ7XUlVg`{Pjcs?u=L=`;Ry^O4bDiq&WdbX*w;KN;#cQ8U)Ot*b`0G^^e-"
    "c9rX!y>iU%dfmLB`ATn9oJOJ1cuGtbxsbG00Ftdhp?y@j|YPp`)r0jvs8VvdVo3)7%Lpe?<*2+0@YpnAf$7E9bs*DMmGJ6&iEaPY"
    "MTasbX7W{pI0CsQ#%;<RhLB75XEGj|q2y|Gg?e_QUQ;L{ExiU?}0ZVwm_8d$8_k@Lab5`($1lud|M=gwh@h5RUZE033rkLwhK>@5"
    "aMI3PgrnGqSg(RFxp@G=$S7VQ+x9WhS4l*<^Sozp`Mp0?|w|LM0^e|z}#x-"
    ")5}>p|?=E~J+#TwiP+x8UF;eU$0XqJUO_mOM1RieW>LToOB5&DsjC+s?x_1F3lWS%7-bya|>N##fH-"
    "+MIa>$m>x2zXjW%hm?P}&<Tz*Hgagsj5HB)h5`eQ<-"
    "!)luxYX4=eB@ZK`s*NfiQTo8?5B02kwBsj4T)!H}(w1xgqJ``Je%cyCaK`77!qtEc9=a7I|$=bOG=P@H;3VH{h;FfU--"
    "4zn2Ce{zxU@{y2&{v}g?xa79uZ4(}j!F{>#6@f@AuaJn}5dubAJ6zBrixStqI_f-NAJh@&{nW@BHd$zYXh1;S-#6a%&&K4ogSot-"
    "S`KEWq9{618RK+TxSqmhhiketm7b%aaQwcR%6Uc*h><pp>KW-"
    "R*V#pLFW37&%H#x$SJ}INypMJak+Ye8VUw{7m^yb6U<46A(Z#3tp@jvi|IB2(j^F?B&O@^c88+@5HUt(9_J^OX<d{htpocE$5wj0"
    "608IXM}p3YtCA?diV1LJW=wFF5)3|qFycH1%%vytw*(vU|(uDH0Yw&ZJ^PB|7aNQf258R=@U=S<Mrt^%*KOd;t%<26PTv^Sl(&3J"
    "pjB%u;aG~oG`o76W9>r3h@cVML48Fa5R_ZH3_{b!1rnJj-"
    "2VohFYP6Ok2FkTep!{eL&V}PkJpAWvrNJ@|s#k$T)nxt1L40`7|nnw76xjv|KwZqQDburBL#JC1=3AX`Q9DJUTCBou4mGt3TSd#*"
    "Y#LU<Q@(oUV{|wN97#g#hDK6|PXo1*r0$s_?Pa6#~&j@VTaBjpvipfaq#h`fH2j*!Ix2qc>bEJ!yIrGm+6@kNbeTG@7XbabJ&qzV"
    "u2Mu6aCp*;aCB%M6s~7>3KWMZn58~!6rV1sHQG?rx{Q=7Q7W{K3SE#Nx4}Bx4p};9o#&M!g;s9bo%*c>FeJFUWxS60ug+<4Hw0LZ"
    "UQt+4NQm0ay>I}DK0mf?_9=d^Odukzul`zN({8%BpV-GqA<e?4%$`JYwDBzpQw~@Q0wAaJPT6|yKYsJosap@32m0&L>q#n!8c)VF"
    "ChXg1v`T5g!_dlE?sGrXyU9M;xO!M3;;LFG{&Q<vTT`cKVJ22?ogK20<n!_Q?`9POrr7>LxC>J4|nR~9+7I){`LYu<KWFY#%Y1Q-"
    "X4V&-O^vn+8$Dw*)kVns<rDhoTuvWVz7%taqfj<FsVVW*~SNGoVrEQNvI6np0h#BcImp6Ws_fOptV>kA;gMJSTz!p*lE-"
    "Z9qSmGRx^+~}8Zst{ytsFNHko)uRe81mZQ~%$E%<sCnOruyBc`?87<$UW~{MHxNxWLRwu)G{`iMOD2^3YS%^AciO{JJ#u5YGK>!}"
    ";>{o%@YUepz`~G40!Y<0V4YPdTUz5;If%8?ONzMLcA8X++XncPHvyLd5iaO^=7H*5uFn?NMi-"
    "X&Tu5wiYkZKBp{ZAvk=6iO!*Ua*I`thYd#}q8F@!zJv^BZ~%ZT<zH+Ze3(0uxMIOH?X6FcbvO-2ye+)6Ji6Bw=3{``f|>-"
    "Uyit6<3Eq|=aBFBAX5`wV_!NX2KOT85!wqJsKDBnaySX$YazB+m;JI(%vg4(d8U|?z?CLmx;t=CbBk*KKKL;n~Dxe=-"
    "qBjZ|#H$jWRYN~7bsFY+egOFTHwh#)M`%b~@C1s#8M_hR!S0PMtAo1nbRZtj1dKm(_nay4$A$|QhSF>Yb0kHUt+0a3zSvRtpavYs"
    "CW?bAG%ai%L!B~j15x1~m@kr;xE$!_O!7R@!Nw6#;}3#JV#U2^ikb8kG(%wd1(aDMSj&dMdLbP-(O5<q$`u-"
    "d@hRhW1p1!Z0hCR*Q{HWGZC4sDd@RM7OY)=SesESoQyXVTVojLyZd&>OjpxY`HV0S&p1jfUKcY5JA=-"
    "jB48GxjCwu6IiPr=y8Ci<;Fl21VT{@7D!^qdL9kE(r!G%+tol>f8tog{WXq-gJhkT4|lp!jN&(E=*uqV{c_lMxkajp}x?B(RkhXu"
    "N5;|?z=7m<8U#t5*fws?)33l0X$Y%B`&g-j~uu;3-8J8BE8+p5f1XNz)d5-"
    "3U5I*2iqBs7i#kLw}yjMLEqE=liwcU7f~i+ZgP#Lz>fB|$QH=pG~;tZGAG^m_NZyW;8NzP74Df5wdg;%(=f^X4RwO6-"
    "^muh(SrZ}8`~V9yToxv5Wg@-w!0lp5mti9q5yXGg!$U>6k>OIWw#$zj2g&$)+cnG%-"
    "bU6zx&<8QC1|9Ul}V2^kTx^ua>{Od30Ti@ZgzObg4cBgNrQ8LtMVrsEnwhB&SwIhabqfdMt7_V*U>(hr%PoLe29$v5CFRvcHJpOR"
    "~<JZqWKfQYR@bvZZ;q~7h9^e1?)92@}fBEq6`r*^3pB}D1_4IV@TYOM(-"
    "LD?rJ%9Q7<=V&|u0QwV^T)^c*PnU+`1R@fe?NYH{?M1}AAWs)_w>N;{QUU*@y-AD-+1-"
    "#m&ecV_=`V0KfQnV<<0em{o4<(xA?_xetmlW@h|+ezrFml_%!~+_{aRI@h|@5*NuPlr}K}$y?VH6zvFu8m&f;SzV*v^{rV%<w{iW"
    "o=XdX(s1>i<qqpnbz5o31I%cb{9^Srx{=~nv>+k&V=Iu{UKk#qYPk8$H_5RoRqkn$;=CAMX*RB_S`26(rf1cia`sw}i+kbg;eGwo"
    "3`tbUP$M;{Jxa{wr-"
    "hRER)UThP-(FwF<1at*Wjy}!^ycyX^;7=x^!BfxetQ1MwRrga^yTOGU%&j357^$sch&Yk{=>ff-5<QJ%;Sd-kM{-"
    "p!27t?xQDmbo4r2$=ciw<AMoXV5Bc)ryI&qZetrD$=}X-"
    "$zrS;yKR*BNYkA0j`s3&K^|@Z&!T10B%TFI4UQR^6JbnK1e7*Ae;rY|MKYV=p`SauZKjeS>@`qp6hrENH-"
    "o5$tx+m{j^3PA#o$&mXyV<|KE&19n+zo$w`18Ny7y0{}<a!>y2S=Yb%C?=~Ll(PlmTP&Y&Z`^dItqM$j_XF@!e9UK_#O@M!>>Pm_"
    "715Z7GG1mi*0_dbM!YZ(1%a&pT4}g9v<!@Eb|Mu(z~Z$p5I<S{o(D;?;gMVcmL&||LH&e(?5Lok5B*Ur_XPnzWZ<g{eOOU9r%~;{"
    "`2FfuRneI?jQf_|M>S@gX=%?;oJ58zF+>s+n=v>;^DX7{x6!%V?O"
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

class RLHard60(Player):
    """RL-PPO-Hard60 as a standalone Axelrod strategy.  The 85-feature MLP weights are embedded; only sample mode is supported."""

    name = "RL-PPO-Hard60"
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
        _require_sample(mode, "RL-PPO-Hard60")
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
          name: str | None = None, **_: object) -> RLHard60:
    return RLHard60(seed=seed, mode=mode, name=name)


__all__ = ["RLHard60", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((RLHard60(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{RLHard60.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")
