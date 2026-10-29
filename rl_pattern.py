#!/usr/bin/env python3
"""RL-Pattern as a standalone Axelrod strategy.  The 85-feature MLP weights are embedded; only sample mode is supported.
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
    "c-maO{myMoa@2V*{rB6}s_!ap1tHX8jK-43M#eBg-2CJ(GIGDoV9B^opYuF>uT_;Fkr9#q`!E0XU;pj@`P;w$<v;x8pZ?qb_1Ay-"
    "$N%wffBWaZ{<nYq=YRSy|Ld>+v;F0dzx>yK{M*0(kAM5e|NKAy@!$UYKmF^!{GWf(|K)$rzxpr#x&HHC{=;2={N-"
    "=|<=_A5fBx;?^I!e@-~Q$Qj=%HIfBRqYgZ}a#{^S4t%l{ky`#*mF(Apidt^Un^_d09*@mK!v>vQe7zPWq+*W-`xAHM!Z>-"
    "XNazcJ67{Kap4r+@3Nw@%w{4gJm2*Nwlt=9%9ZWAkV0?h$|cy?5h(+PC<_@owv9^m)&Ve?GT<=DugOzP~+gA4h-"
    "G_ImrAwZ8WDo+tle>uc=0+Z*G=fBW={;&-0b-&*U<Z}+Re+1K9ZJMnAAH}B0)w8x46bHp!;Z`IGz-+XWF_@--(*2arp&)-"
    "~Qz0<}SBYu6?zkWL&L_D)EeolOEJ&hLsrq8)|JjnR)OW!fRWc;je#=rjd*-"
    "bxj%=bp$cg;6v?@zxvzTEe{9uIHEgWQk5IOpi!SzkP;_&r^J*5m)&tF8F-Gd|eJhY|mBzxbQ+Aifz-Q-"
    "65&7ms_r@4M%|FaPkx&x}tP&-eRIeA9f9@ltO5m3UF_O@E?2dYdbL@Q8;Mztv|Wer=0~9N+1S|2lvBJUohT?9<{+=40mp@-"
    "s(#u6g6j>v^2E;)BPpkH0_HZ29ZspS>P`8vpynSIOrYuW{`7-f>9df%&cXZ}c0V;=8<`?GG^H*Y?|cJZ(N}_uq|I-"
    "dp^tII}mtavs``({SSN#@UZA7zZk!%UJPXdb>NmNZg71U@zWPysvoGaRgVslXwMt#v8ivD(2-"
    "6<D>2sFJSOBua86T@ulKUox!ue`bhCx<0S6g=Hxy2eBW=!7vj(2$d9Y%)Z%Z(`H0uNzw_OAB%F~r+il0kxpA?M{wRJO?>@eHe9$;"
    "4`8f9NYxb|THrqMAMO^6_?{BYo@^SI<Pvc4UcqS{J$6W2@54eq+5|`@jIcNU+_+)3@apH`|`R{)79Jw9$oktX}N=LL!u0@=(2m?3"
    "%dDE4d-&^tD#C?gckw^Kq6H(!EzF$1g;kWkjyW+3Lhlo>_$0pt(aVla^JaeDBID?%RAOC*O9WRAf5l{ET-HP{k;{uGH52QzwS@E3"
    "Xq`u}iej*mcS^P#kq8_g#uV(y;6OXs?WxtkZJ_28y=EqTt$5M|dzIa^u^SDw?#~?m&oD&{IeD0A?C{FNcCti1an>HNe;`NL;?eVL"
    "7eEodo@nm1zt+>~5DBl^5QW2Zt97TkSTYu|c$AdU)$Mtw|xP7l8YDdtE02YDL-"
    "!Ov5xB83s+eXACeaLveBfiNVT(menSLZikW1NmSW;&JmB3?xHI1~}EBaB4g)#<+xBH{?f>D#;eLUA4A2*(+Cao)xq`8VU?6U^cw#"
    "9yyV(Ox8%c!~RP9y-4~UUvMr(RQ4f5$8~GoQD;sB0>i@LB}P|e7vrBo)MrTOk|K*yzQ-LwqEP`OX4TREgk!dcO5_GKmG9^^*JNjk"
    "<lWh#oLW*s$d-7Caxr3DxQ9v1U-(30rA5g$uSZ{KJ*a>@5Rd{bli-v-"
    "0`@KoSOLG_tO!I&l?dY;zYdjxR7~yA}OwObadmf9beN$5c^1O@h(=Lz_`X+xin(Jn4|t?+=Oq%X^XQp^daLqG;(5O)5BdKnd{;xb"
    "J)K3ow#V3Z)RM@wQk(H2<3UEBf|9_r!C@E+<|yj`rQ#r;<Lx8jE5ZmhBpx*B<?~)y7;-"
    "=AH9joM~_DqcRQ}7?^ry?yW*roaEM>{>Ijned2wIlf(WbmI97abZto)E#F5i)k4VIkX>lmyxZj)iCtg_`u*gd@4tCG1`Z&`OG~yM"
    ";$Gz84bi@HZak!{oULHbDUu5Noh7oCW+~Uc^XSzBzMA-a5@gG}6)i^Kl;^GDB0Y~&CX2gSwcfVXq#A%676Yn*`R}?Q<F~zI-"
    "C@$KK%0P#DM!68*Au?P1uI4B7BHl%5)fVBg`Kgi3+9IvSBR%}%PLhvnA9snAwDlh5jl&b4BhElPynHtCpF3qvJi7Q7@u>82c8iZk"
    "mL!~1brG>H-"
    "u{U|5_jn4yNSCIhh+1tw=d{Nj3q2ZyoqaeGUdgGzVR`3KXLDlv+Io<5S3HBGvZcOiE**wJ8+44V)5cLKy;4X<G@9{UKM}$=KYaNI"
    "!Q7kX#CqKCE}PzB<p$F25})0O`M_lK5h7g#rNqiE=wG$=g1$ajR1CEk64=VoWK_0Y7$GH^KBfCc&RV$R79~j*LsDKKjLq2FdsqqW"
    "c5azh==x36-In>RWPD-jSCU+BtKQ2-"
    "gr}SOXHA7;c|VZxrlLG<M18+#>ybt$s<v(M2<UgyXs4FU%wtd<tb&xsZt)Fk+eI9P5C~a)r-"
    "d#p`jDW^GinPj{q8<@WrRfgUWB@pv0++UrWALkr`jG$HR-OGl*MC2&9`Rx~Qxu3H28{k#~@E_`1^-"
    "M3hL86C!42t>pw4agHjNZlGsrAD5GC#!-wKDC!o)^g~YgsGB25<hO}8KroNk$kkobHD-"
    "%z6F)VIr8x7=r8y_BQRPLoqD!W<7AJGku0$pyHS1PH<UbtG7Ej}3;)-WYbxvg+$5?46KF}r(L}3_zsE-"
    ")U#i7!Rq9V@Cm%oIHI1YCF$Os~N;c0r}o$#{b!bk1wT8h#=GE@A&c%(aTedOtQ(o_mjwsX3in)Sw_D(x!%TwTX~AWX++yd&REMA%"
    "Nz6d^SteSYtFr19|w;pg!sBf*gh&(v|IL5N6}fr%>?At)a6p&+l2`UpT#FT`EE&F3Y~D$jw}(;fZ4ySVp}CL(IY*X-"
    "^VqNI%%8G&p@;aD|Iqpi6dkd?_f&R_gl{NqM+iU5#VJW8H81MzPoNcj;*u#cQYj`+w*El+w>1@USlrbZNNZW$wAeEih-"
    "+wp;ypJiOJtD^VN&E?(U=SKxck^f%H$|v3$Eg99VGsIK!i&q%0r`O$yw?ahZKhp{31Bw>+64iN^@sHdAcVc-"
    "|IVa9s&nAcJf)7r`wK?!k5pm9_Jk&Spv~_;i;oZi0A)V@d$CHk`7!NEGMV?&}LllZp!o^ePaOw+2UX8yo8}HI_c2cCqYn_x7EuU?"
    "?b*^w6z9_~bj^^i{^ulpz;^IW4J$_wL@kF-}MbD*VXjLZDuSRDS*Ega>=GCK?@5bedHYM-"
    ";ieNdZiYHm!F`k||zA>jTI#Wjsx}Zh$iVwd?iu&FW1_n1QDsnZ33Rm%(C%tz(yy1g-BgVzwAw4U@=}Sh5O-"
    "&jvDgGyNX1>}L7cA<iIHSBtLl&EMD(bNai;>{m8__{^YR1PYANlX2B97`lD(o}(3>k(9E!-"
    "O52nWkcqKE=8DwX)ESwaq4j_BSZvFV~Y5pKe66kUS`<vYG(akVEgHF}mv9Xc9B{LV>>q9p#FK2Ut}IAw9CBk;uOei<Gjq0&D^ogP"
    "=lM>SqS)TR8vrBHJL8*h8gUt#QWM#=lo`tj=X595_Y1xtF0$n*I!19-"
    "$0iAQwfltk_4SfEG(R1|6E%L1HtddM}AprZ0}%@<dSvl$02Qik!Cc#3goB0>?el?kbFX|1B}j4FJ(A&ra}fh6*FC)ee@2V&v~Pr)"
    "lbMe+Z~4<eS(BI^{LMF8&<K=B$~z3A3}W<-1dM)@OOPZSGL<j3!i%3$bRH%c+ut9V27-@1@-"
    "!#9sXhtao>KtV2wp6mNnuhHjGnn&c(&8k1-"
    "PEgUrTZ^xd$W`2$&WDfR&R@t+2^bYc`=n6nzK~HmPz>wW#aA0HH$lSU=EmQPi>WgdwN-"
    "@bsP+!|E4$1H<Wb#41ra~+cM7QYK<pwo#<RRtqeqmFPfdRww<|wnbVYY4P;%EYV8t)xA#f)svhpileO}^qeAMTM9%nKRTHFu?E`P"
    "@Z3>eq55q4+^^*5sji3mlMi-&TugNv{&=pnj}=nvh$bwI&5vZS(zXh&y=>NhS1DSN&weFpa=E@@O~k#H-"
    "vb*hswBS6I0ot3buH{<t5$d0VyIz0-oco9ARL8o+di_F0!)J_UycZ$Gm+@5%TaVa977FasKAR-%)G-"
    "h|R7@s^&Z#)qKPhvXp9~U(}Wo#5NEptHUGXlpHVioUtfaAp%Tk+g~*DMktxI(;s0JZ#_{D(*J{rE&9!3z53P8l0#IqTghCT66WL$"
    "?>j*UV;%@`2(is=0VBnMvbcH~p+g0C#4i7%%7JUp7#JL|G$LaX%wpM|$=1<=pYnX+IDDeOCV52YSDuE0s5w?lAhJ9-"
    "(;qSAR`4F+E=NV(nJn!lR9zB@TMLlskS1Ae|6k3_$a)GBmpIDBj~{x4b<I_!)>c4(C^mC($=v>ZLA^*pQ(fQ0wWIMe^2RBMVdP#v"
    "Qrz9#t&I`;R_|9x%>cJR}#(@m+|?6Ik;zC_=rwsAp(gryI@qL3}LIBG`)Co(b4>dEeAY)0G$fQuNr7Q>Q`iMi)p&8^3PTDd%sZlV"
    "4oH=)Jm6Ojo%VzlDk?4$1X-yHBw+F1zy=I6T)qp7Y@a<S&Y6F=iCbfEnAkX#|`t&}clpyu>S_fQ>J{gqnRl!F5oU6-"
    "Q%m6<_n;q(Y?V13B&Pp5tJ|U5PLeft)m-;0>84ikd-ON%S)=V`M6@@kn`5P#e9B5JRids7RI}1CW-uf{{8S6VTgaCw6E(ILzP%{+"
    "UsS#u*+!2Q-"
    "CQpvEJnM%Yw7?!Mz6gK@{Vj4<$AI&K2trFV{}{_;W};z`7H?x<TFnl8pHbnjFi^LJ!`J^AJY>HItqzvBo~U>{c;5htT5Rdp1ha2s"
    "0YWOiLOZejJWQz+PtA4AjU*c?S8`8A$fBR#reS(Kb{lpfGSTRt3d2<bhdMBzf}UTwiq@iZ8Yyx~TW(we3sjyR>9hZTppL=z|)Vcc"
    "PN@sy)w1YJCa`xSKKgI-XXnGv5v+vx#5VFVf7S0v*GWP2m3<uM{Taz1F~qMXQg78hvK`aO*>3PxUx72<t$wf0$?lT~<c`d~)!vQ&"
    "z19B1{Le?s6NV~2>$jF?`XH-hftWq@fYL~d}~$LJ}lo#pV#>ph?gDhY;qIZkT))Juu^3NEA=;Ds=Pi-f&0RS;!_jZ-zb2I5^sdI!"
    "6zXcqYZF!j*x#6jHd1P{0|_-+Kb_)$(4bSRU;Vc%>GK&awM$8#r=P1hHZa~FjssiF5e>Tx-"
    "z=6Ug;Tys#KQM|^BU__E>J3b}l$w&KbOns}6o=iSC-@Ur(J>q7>M*#j>K9~gBxEGOa0JVI}Ba-#_rV-)eXZ7D02r%OTkX-"
    "=?o`ZqY5OO0@o}5R#NP6+1m^$q$8=|7%#|eUTzZnTPUL^l{Fhr|9KQ3Nm&A7Y=u=ZsO@c<qsPc05+UbOh+gz3nkDg=!g6BFZAuDI"
    "m8D(<03KrI`Ob#_lU8AL{bPgfS-Hv6O(PrcKh#+O_MedF;z&@rIw`5WPYV8_>f&A4LZD3w3)IbXpld>RImH21Scs~|?SC2?9iGb{"
    "m>A1Rh1^NoT{yO316mZQ{{%^_tcg$mVvbrf-JB4b9>BING;*wnsUuc*72dU<Bqx%Bi=@r_RZ)ifZP3~&G=5?Lbvbvkugn5;pf@@8"
    "5al|qEN?SK3M@B<9Tv4|uwT?kWPMeK_J&iBs<#soGV7-"
    "ut%^vFu75xh2)d1Tm#a0yFOn=}vs01}^R(77|Y6~j*{gLNFVNMu(KRbro!)8g7O28=)GzKn5cBCtiu72-4sRR_q59%F-"
    "kN3*mH7cwu3(lP>Je_7<kn~DOH6Zd+~v*Xmzj(4yp0JwZ^6M*8xca6hlgclr`dWf7A6=DLa6t;1aJDBdAPFy?A`-"
    "!&|ftMM1_U>_*zD(`o{D}ud>#Is3s-*~e3J+0-Qtm5&#rtt#9RK%YejBlyiaXI<+Pw&@Op5_=^8GR-jfY7dWR~OE)dRKz^E#_W-"
    "%$v~og;z6FK{l5$D<xp!~NwqL02&ta4qIsZmJ*M6}9`PH(9llI@S19i|)<+)a4^{6Bu>sj>a{dah)l;=>0u*h{}b04erJNN;rUlC"
    "e`<zM#37XNc{Y$%`fRZ(TPKY6b0ILu?(QV=OOBnK{JFzkl@fV;3tf_<%&-"
    "2Y@NvXnI56XmAj0}NfP}Q^CC(fCbJRO8DKgMG9QkM$2_c^9PIO*G@T`#jg0XsJ*Unf{mf06vhh|(>O9=m3s&Kl0+AE0dq&jgmk-l"
    "w+Dag~KTI`P6#C2z=kUy!p)wSKsmi*hH}QMLk$^nn$@t<T#4(CjDa59}kH$=}0U~x~VS05PBT7cppVNb2{`n!60RvS~jei*zk#iK"
    "W-TiC)qijP|T)X&hlz$KiXjB2qJzd@)7@7``f|7^YE|@e?37>qyag^0bMV%L+xs8fBpoNY9mWDkqef$yKGqByrauWcSV&+mp@bUg"
    "|j}l>;8j!{!uj8QzYMk~-bC~~{F^JHKLk9gZ3J5gc7P51TAQJxsq?>9kl6|HGNF-"
    "4~6ZIek9S_S7KM;^9UXufG+~SA?aFXCtY!8(#VeDkNksm!mMm+eXPug$AA(c#vVV}^~yvu-"
    "2wAeS^+wj~rjx%q9XI7K1Ml|LnGI)KN{v%FY4k;7g9)GMNj-r|hA)a#7sX4;t@-"
    "^C5xEO|M;(SIBqz8&THnRzWVe%+{8G)}XHRHs$2(Q$<8jziwyVA675g`~~xw)r56xc<=UP%&)_ti;!jq+wC*+OUP(Tp&GdZv`@ON"
    "p|<L*rJ|Q1VdXegay*>b1aRgv7<MaaFl#>h&N8Q*}f1apB_eQ>-"
    "n=$sN^!s9KG=VL@KtMG>kx|Et|;ZWOicivJP8=Eo4|UDdHF!5kTUnPg$pv{{2wj(3ECsA%Jd%xe7@PRB3io3`wW;>kiHi0~T^ttJ"
    "zLUlivrB4iwX2Lj<*pq3;kKAlrAFmznz1)7vonJEGh181616>*@vk2w#6`Ur<EoT+-"
    "_e8ede%gNy?if{$)_(29zJGo&pIfoA5k{=|5M&kfj_pHO9sFCOfb2#6d_#|;-"
    "BFZwna?LKTMm*X0Blu&+uQ#B1WHyL&x6ryck>Vmm9oL1Bs*%l#6b=^`ty+*S%0eVUxGKq8;iEI*2jaik)lhXnTmtg~pv$lnNr;EY"
    "haIn)Mi@NtwWoUezs`RE8i<Yrj5AW#KmOH4_Xb<N(0k(UwLE`|x+otU7-"
    "e_W^1uu`6`QJkLjp}b5CCASGs%_bonAWjJZZ!!Mt|^JvV>Z^&Z{|}2zFU&QEXoh1T%PNHhQK(G~x}m99J|DbD|m_@r^9hAn8N~qD"
    "OjW<xu`SKJmv(tO<GK`X-"
    "Q$S<Q8~8o5K%4@RS(r`IHzNKNsc;@S)!skm5BNtm%O^1V(E<DdwTaS@;4#!kU^0Oep~o!2WUMYM_QMIL<5<NZN8r(=rG?Ez&MOAA"
    "Z|l1t4|AyqPpy8BPuTIXmgh!N9ukKk5rO$_6C<PRik#V2Ms2-NFpCmzMpje-"
    "3*G9MCRBi4UZe~JG?g9OT$YQ?HnCrgc~+#ioUv3?~}PI5fRIQ|}yaA@II#zQ*hgc7*A_;!5g%sv`_aRVnPbI(M>c&_oGm?b@9!3y"
    "dmJ@|KVR@{=tuY^boEtE5GbLJk$nLz`Cwv{lTq8GpyEy>D`rbUteL92_B9wa;|D0KYP#BfysG(7|*4@1tPUDlt$E8?oh?^cG3mrh"
    "n1!qSe(R}paNx1+F#xSi+WY!Q;;H63O?4jfTBbYj?{74&fAQMzn_vAx(vasD7qMN!mMY<WxtlY!&fi?G9}faxj?^GjxhK+CxMM5g"
    "%0S%bz`1167O!UJdqGU;uiM`GIXJQ<-"
    "qxUf_Cm;^J&)wsCKOqivaRktD1L>b5&oOqVuX^5)_qYh^5ENBSA3u6@sSjG~;5u=HXgAqr}6Yxgum}pOjqwL}^DryK)L#>`0Jy3@"
    "DGiHMtm+G@Kw*pLM@YR*>BYZQ(qtG5qh4aP$2Y_~7NONY;#R(FRC|cP_Ajx%$_ovBKgmJ*$oIb+%YxHPT^u2y>{F}v{h7D%K1$f~"
    "=q=M9303(g|S!xtik!0qf+u53FjG9gZ2W*Ha17z4y{gH$>F$-"
    "!%+irK__|vn*<A?xV6*WVLM~6w9pO<)$QlR_LK0+!qzhRQnza&H&PhS)Znp8r(FxPJ6bB$jVO?s=2J6ecOWpLc%?jkJ*))Ck*fV5"
    "vE^YSP*qU>S}n<EsaSdmPl{EGKIN1kA=4cU1U%5Qa1M=Zri4$z67>6MU4HO2g@NrcTo!ls&v&r>5_as?w9`pnO)^iJp-"
    "q|a6~E%5Tg2$Xs$oA`(okb(IlfGG0`;*`QW84#o6#v~3I*92G+t|q`|(v4`2;5g8YZ<ClK8;p7%uxJeOJ?J6p6D4yc;Ka88dVm5o"
    "c0qJclYmHqi=qE<Bf>~N${I$)J6ju~qj)>2vU)PICq$=xstQml=3IUv&-XgPfYx#0NMq4WIFdts5&*T}$27m!BUS{omFK}}6A-"
    "cCYcc4|JO|r#!zuYbU5O;%F^h}CIG_MkF}8(-bMKa@q#jIDEpMoV0M|Yiix&_d3#wNJ-"
    "*FiMfYZgZFYyIP=YZY2pq_Xd6R&VUdEdNa)zn$6#?K$X5)}K#@WBN%j5?HgyBpLvi$vr|X_0lh!4Lqucv1HQ2zdEt9I7~MfGl5<7"
    "UNsR=jF_Ud^n?l#Y63i+6Ef4&fcdAElwkaw(E92;f{(P^l9a+S-cgd_R-^YIIcN?qUr*ZWTaxSr>j8j<l2aJ+lljt+!R-yL0>bB-"
    "JY)XhxCDz;&7~)#rULAarZ`7tQk~~m$9dZwebRIP9e2*^BSVTjPHEs6ESx+AC~xAjG+cHlRsxMacbiQMih9jKqc;42MiuB8)IsUs"
    "L-h<;QyJ&df@;)n*a5iDxLTS5n5pNJP*zRO(TslE@9kb0Sm+cx+U1+voP_kna6-"
    "r|7v6zSA{;Ki$K>a?oT3j<6AS?{Z%l`g~8ef6ZBclB>Y&YU~z1WWki#WG-OjML*+7qVqKAiD1}T!ysB7Fs*ldBX~dw(<ciS?=QkP"
    "@fph_JvJBJG@+%L;BYhSF>3Pu}h)ec7Qicj0zaaw4n9f%<raEfV+#YNZ&UNQcGSHuD{^;c&48N={NjuP>Nva(DVq(RQ0Zb6gQN!>"
    "FQ9uc!sM&|N$S0aoK^c==41$*Wv-"
    "l}yL`10^jZ;&488uUl4;Gib(=1WM<?o^ViFyjKKYn*%==A6`hti9zW_0*4{O23?tg6vH#Z#nRejq01>_LA#@cBD0B;Q~c;>w?^%Q"
    "U|fRO3)HL9)#Y&w%!U6VQ<wH)TkmdIHJSw+hqg&=un@jaI^*!8ze70&y6&iuO#E0A10}p=M(=y3X|!&qPv2<&gO4A0nrkMiRdo6g"
    "UcQ61r=DxYiGO!Xd`CEWkR?kTQlS$@R|c8UH-4Cu0L;c;{nuNnFvs^YKQ_a3B>n(DgZr*#Z`R#6geH_&j3}xW-"
    "?lr!cGRNu1WnTBPqxi4bz)Tf<Y@7H>s3q@j`^Lm+Z;jHtK84jCww@?>zu<Ci~`C_Y$pSxiwh+x(o(yQpv-swLIPs3Y_xDoM3f(kt"
    "*g$T(nNf08QWtm=||r?hnu3F%7)5)NLe`{+)Z<QUFp;DMno^emEa_fv~j5&|WdQS@EQ8omoPedwD%2BA4wL6$~W7e&s=JOug?ZyG"
    "x2*gv5K2QrQ;Ts+dhpBsHp6ycDP<Eu6o6AMlPoNFRn#*yep&|*P!f9`OZ^Jy-{;hH9FQ$+w&evqPuMO#Nx2#B~}H7{`e$-"
    "T$fsm92(EzqNoI<A`IF=W*El7W<=?cT}gTt<Wy*5E-"
    "%<`@mw{~oGqq!jllUK4eDG8_g4aFhwq8sgVxY~vzOoG*<j^0C3aU5o_47C*<p2gJtQD}Ky&?~)9=c$+}sg-"
    "B2nQ+SeTlGTVBacVRJBwz0VVTvd=+c3k;3HK+V-ZoK}7H&g-"
    "q%s;sT8Nj)6u9Z4Cw@ThN^Jy#i}RjU$Kr}IJD{k`Fv7Egq7e@P$yrqhUBDjr%x%TcgkRZ!LWTL{xu=On_3^2U&Y~PRonB^&>08ZW"
    "97sY%gXoitbig)9%;YiXJyn;}p2sbucJW<;#KK4v+#ZI!C-"
    "0GffOW(GF3Abm6+9Md!e=~{3KLosk8eOL`Q!WpDEa~{hceU>c1y&Ex}|F*NnY?^W=Q)W7VY;HPJ*-"
    "vJ~54HDia>1>lC>e;z1_w2$}pP_)1KI99ka4w)l?dpI%C6m|ehli<9y>Ortyncx9l3rqh`5CNNIs;*=wf$S}I2uw-"
    "0VmGFVY(<n!zO8X;`)5ja!n!cFXgVaw4UDJqO6d+jm@n*=C$rYM(WedUyI-2d;n0gDbMC(7UUc@{Y%8}$R3_cfx5tfk$!l3x-"
    "Nf5gZ3JD2%&C?+2J!zH}r#ENa#Px}am|pwtY}mMnQzI_Onx=rq&jNhi;t)NL85wPg_$+0^E}{}NA8~Dfp)&bXaW)_$2JS6q4wE^#"
    "LKt&(cjy4@n;K_{e8D?}0!$?&T1FmdsGSfsArnq_9_XZyI7IS|ob!2v*Fh#WGhAL#!8Ekn6n?^~tertd${6HNQLP&LA=MZ{gZZ1L"
    "qfdeshUU>XIr2Vz_k>2!e{ctx3Ot?D(eu)ijW35PVlwEc(GC!*=mF43wL)_~$Sz3Mn5b6A2B3ie8Y;=^Km806^1%e)c`z3*hqot!"
    "mVtq}#EDyo$n){2zKmN>O#pqQE*yv<SX57?03;v~4Qh@Dte@4}p-}`+kN<mSAdX+K23(YLd|4R|aqcFR`Bh^Qn1UCeRr3p#h-"
    "jV38;tv55F17!PHd7Qq!u4#h{qCALeDjmJHG1&lt}>sH1>z}B(NC(Uf_o2S3$-"
    "&8MaD9nt$(!XU;E28e}pV<Xb9eoxOY~OKCF>HWbrgaSyx~)DRp+&wk^=T#eEn`GaR#YC8l0ysO*FTVOz6<_mY4hrJzOq2xX?Alwl"
    "X{-CSUE24Yr2Gk+GkXwKuqg?TahobXfyF;}ss@4Y+7S;!()a{`zQ^+OhFima71r6ol|K1<7{XL0-"
    "<9mvVZj@uvnnkxZ8SBlAkdetmJr(T*WbfCJFdd8_a0n7f{-"
    "HRfe1lD4r3zF^27nAGUtv2mJA)erSL>GAp2C`1gAkuV1vMBWM}(x;M4bqi67+*VHH(rOP!f|P8VHcJViQGlMC*iB=`+e7U+V#9#v"
    "f9LWq(B6qRzw5!rxo)50_3u{8uEt34$nSJKoi(w*x#+cwI08b_szH*+y@;`DVXxLxai?SHt-"
    "t@rwEhn<;Y7myreVKJqsx$mXYKVp}Sc&?Ry>U1f@3N{K@hyQUc-VSgg40oCkce=L}L5wDoNe@UWId?lZ6G*m0gSz(6E(I9u#v!^m"
    "j3Ul1ncdPM7)6h|zG6451b|fo$3M;fkNXYC!d==*?n{SB%K>m&yWGYZ#`5@zXqx7^)uQl43l!~oe!Ob+4!UqLx<d{?Xb4Iuw0b^7"
    "DH{1}?AV>v^%Gi`9o+6_chHu08;SgrP+>A(?(FckOm>C5b>R~@+jWi%3(zu@H$xEaJs+zMk&;7%{CrXsV$s-"
    "Y%3ybt+s_=slYmriR@09pepmPopqR<o;c?kk$ykNxgq?%te1&f?eUlApnIu0zrU@Zb{JkcME0*xumYO4IAg(5v;^rzN))}(GxzY_"
    "Yud)hNtj_5P4cw=x;IuAI}ASa0HtDdJsjuTFs3mjPTg~`C@GR5z^3Ux?^4LMQV=H<ywJdHR^v@_5`%25K|E@PC(+4V#LB8$Wl2(q"
    "S`D#(!`EmH$`H%#Dj(VdCl73n`6ZRiOgp>kcwQhD-dlp0MO45LbT41l4Liq31TCIcfedY0(fRno=LZh*^8b%9AO5T#ToZV@w_jwa"
    "MRSr^PIq7Si(*al2AZfWQbu)zUeMd4_{GF{=Pag|_|?71R_F=dSSxu@L_q+r6O2eM`2M$pjkUx^gQs~g2wK@A2w7!}F*%(j+d<Wd"
    "h^pZGW&2tx51fb*T7hqj(`)j(V<L4rySOx+k1?tGVkvW+?LCXMDVL2en}9633o9P27(#+iLo5oYtCu8bjW<A4>`fz`o<3DA~lfe8"
    "6uLl%nUW)$lK8p?yl4Z4=k2cqkS*F>Rsf4uj3q*?_;rVGvi3%)BUe`S)Bqi&e(8xFzvq!;ld8NexiG%(F^kqa<}8w}+y+%+>%c@8"
    "L-"
    "#!;wt?}G|Bp60`y#7zn164z0n@^a@xT)^;zqR1F|12We*_v4R~_YNe<cak*Aoe!vwtWL7y&(;pu>1b3m3asr4frMxhk}FiqI3T*v"
    "+7ymTK@j1(Q6!Ha;dvsuIza2pcuuk*kE}f+H!ylEs|WJrNAJ`SVY|hLJcTP1gY^tWfbW|i9Hk1=-"
    "mH*^L?{rwfrj<I8nk@09kBDv%x8Q~RREdN)=i%zI#YGq{uMny-lUqz{CE0m^DwABCc*HUFc<Gr?VU)P%S9>46aoY&oYUlp`V_2BS"
    "gJ>elq=M04=JpfBoH*=ZgkCD%&=TcTEwvdSb2#|?UCLr<gJ@v7gKGJR@pC^RSi7CJOF1C{y|2{MF+<4=b{5~>X2G?(f1SDvZ&*LO"
    "*k#;1ry8uG&Y6FHp?QViDDWb2Kp*AUnlr2Zw!$p66oOFNqd5Taec-)=+t4IKn}6jk~+I3J_0UO7V=39JptIFELo-EeoVi&HNj@Iq"
    "DVv<A$n@Xayr5$eLf)A&*%xDRO%oU5eXkhCj#eVL$kWG<T%v0@#vV&EE6V30?e)^^E31GnL5OCl)22a#*Lzn@t~9|n%#rp(4AeLj"
    ">q)Oy`WBS=7ryJx)J3$(jT?TPQV;`rr=VpS1B+6_LyhEs(>Yzh+~AV25h!aW9FwJw!-"
    "FaAZjhS7RfDJ>S*Y*Tt;%UQ=vNPn36~YA!2WhckH}?ux&t9;cuPm&Q)|y+)d)E{<!;4BT>oXu90V1Mi5i;LsU$|4C1TXhHcxIEiN"
    "B14&z-"
    "Qyv3+fE<K`gBjoe}AaY#hfR89Ex(j`}%8m5#$%;ih4=Bk9ls!yej2Q;92pC62S!t@Bp!CHtDT4^6s8f=aek#*QcyZB{Z440-"
    "(2tiPRvpvq<iv3dDeRc4>{q&ojSg4dK%B)aOd@zrxTr&IoL!&dbcrYjI1E`QQN)R*3nG~P?8rs*jTTS>|5Z+VP&+YlfLi@tVW-%-"
    "!Mq_m-2R)5JeZ?#yeX*@8UQY&O=|+&GtNK--J)Le3TOEVqL4+My!fga0>mb`APFeZ91QajeOoe&F9B3ep<t88+x@bX2E!k?$RX6Y"
    ">xbtG+(-OJGhnZk4x@rZ%$&_va}TLOFnalx0b)RGMtM}j9<8+O;E^Xs7%x9SIN-"
    "a>;S0TfVLPz8#ZpW;d%g<lMcBX4@U5X1#|Mn_fsEY}k8e&My%xBvXQWSrZMlaHyW)M4&k!qwZzsJL!F@+j!7Tr}+Q3&43oYI8X>A"
    "yY{7#t?<BMKWEZPgChlrLf_>ydItTars!hqof6J7?-HIfpsH@*k0;{F`BVGRNXf|$L&-"
    "M^7dr43|Q_uP3$av?2UjuLk*r;Dieump=dGJA<Ncy^#kI9z~VCK><-"
    "C;3B+KfZFSo1k`R5t)(qKjl4Sdb}UnsadVe6r}4Tz&gux&ABcZ2(QZ`aELw)PBortuOx3b?I!vC-IL-aCJp?cJIY^U%E(3LdFnNd"
    "n;+(-Ty}f{dXBdG60eSOLHJ5Ol!$~+-"
    "8_y>`&X$#HUhCCNS7)edjN_kTW2J12e|!#UzZzz<a1FRF?~8;J<x|%R)kDE40{<aF++naK;}Gdq4fU_AVu6|J2-"
    "Vc(oO?WY^As}j0O-"
    "ZMb@4E=1g!U>|?sT@8_Ac=mHEwK$tDCUQ}4jsj0jz=1CVv$UoO+7?hlvp+FoV;5j$Q)swWyG^(4*0rrUl2>p$DW9w<cE$@vsdl7Q"
    "Lcg3x0Oz&+~Mmr_5KDe{TQiBN@3zsrPex~bUqR8}+<oWXDz(~O03If=wQXU$`6FLb7c8kCXm2`^imA{4L0rJi?9yfqu*CF6)y06f"
    ")T(=1cRQ+~c9moX^Ru#TMUjb6|`-"
    "=(@wxq@0r2RG(AlrDQ9Aa>ZoCec`Ru>q%rUfR!ly%#{QererDZx_HU=HpDPmVSk^Nngd@!{IC1nA2aiQ=fi`HKeQdY&Hr7(fLz9f"
    "D@_FJO6LmPBHj&SK;dra$-ZL>~(ALVM|rUVG)O1KIWP1Wv1#0~Mee6v-"
    "_TYQ6L;3O%;vAwK1;{}==I^2kCuv7qolfgiR@Amv<1$VS~AY~(;7iP06I81e#Vqlt-"
    "VVJUJ$tTGgzOgkY`iDk@4a}+KrFGIBbnmQI2g>(h47C2~^iw>S9DQMWt+)4=zSM;jor4r9KiDk~g6qiVBb4Y}NAdq-hq3WnAIv%d"
    "PitvJI)_%aICHoDyKFhDNqmV+0h@I*&(~%-~W1!01fTWa_&k+I&5Db^sY48(0B0fU=%<?Rd@fGwJxM;T=7XUBGPohodn-kM<V-"
    "W+hpx59H%?zT6hYywU80vM1*1%~loN)PyZ}n`1kcBEWGr?AB+p?U*jDtB9F0P{JDatH4&q(G+8vIkN!Q__6bggN}EE9e8ODRnL#D"
    "fPUZcwPM+z!h@>P2S!6`RcH#kU8Gzx5S4rjQBY$$Ujomna8Ve|A}I{x$83h}Q~K>lEdbYMLxEdB$1B@kn8<Z*}uFT3HA=rMW(35z"
    "HuaY>{iRgvrl66@!V?9aqJ?&RmHl3zI$9&k!i*p*Te5?HL$9+^L)UV@t?$4mulDfN~7R`J*u-"
    "5vDcM9jq(4At>h|I<_H%r>4S|uL54wbEuR#jo**PK)ud%7^_W$FeYg-"
    "+D5s2aOg2_1>9VoW*C&YHUXuAUo!jO)lm~qyicwuM}D>LH_`%zL;lJfipa`K*Hq@Kd<Y^AQ4Mn2`B^YC2c~`fJGv~*7v-"
    "4<sTB^e;gti*jSA;>4@zkJxzs>jWv&WshIWCbp`cV!M3;jJ^@)G?W#-tRm&eU378^ClTWU@zQ2dbi<X}t^bo-oxGD2wp5Y6rsOJX"
    "6eqsf1In(xDqh_MwSChOPe>EQdotZvx@G5SS;*trJVDl({i2MH#VoSvO+r0y-"
    "9qV9JM|I}C0rBB%lBsR)N#zllmFP)pQ&X78cy5??s-pLdVi!D(jn*G8*W)MwER-|pIb-"
    "<E<Ic;Jua4ZLdT#zi>TB`?$NT&`Ar@&XzYN%ZN0{C&MQ$m1yCu$`jD1~MVNt{hWv-c;)eAIa@z~Nqq6?0U~kR|=g^K-"
    "Dnp&iAzGg(2@W*ZSlJf~Am1v0zEUmj;~mywgKC#~><TW;7J<FmmL7?5N0#UedK%z>(|%Hqo~P_zM9;ltj6Q@R{XVP?ye`9$+s(yR"
    "U~7kMk#RH;n*j7}iWMc6->&F570lZHZVEQ~rk|3qA3+=jVrnuO%&Nm3=$As4;UC{{20<^!_?&AuUlg{B4t#boH=ju$5siKhes-"
    "?E5UUB-M-{vq^opRyQVz~v&hdH^$IpUMP_BCiw!pD38rl3lXi{QBrpP<eHnHEraJEssoJc_#U1vh-"
    "t=%~UUQmZq{u^BFmu&*@Dw8sz)<d0lJg2U8gVKI=z7p|=1o5(_?AlxRuieKr~A7EJvV{V{ozQI``G6uOPm?UlaP&N4DxCU5|4+H("
    "MlI{0NOHQYSliUdEuJoHV+0Z|~vz^<!lDF$~;sgd2xR*!AtY;)9bPIqUZ)>r{hc0aQbyZX|B^*CB|pCp1Tf?22)C3{`?10YzC)u9"
    "E2n+k&oaGQw?Vl&S)hj@LZsxGU-"
    "K}x3ONB@}T9qwLy@W*T<J<wEEg95H2+q5hTm^v{`mj&YVL>OOJgshmUV|rw#Z+t8Gn2n@ZZV6JcY)J~2v(b5CWTCr`pZs(5=jy<#"
    "2Jv!LE-PoUAMI``2^dF2J|R+4`lydOEFyq5AMpq_ZT3tDtOd+f{0P^;54N+cIm=Ftn=E=8PN84>k;uuj+^eWIjPAB4%ig{uj4{Xt"
    "zYr`~o)9@s*|6bRMDV9^J!n6p$jddgf(u~_LIRoJ{xq1gi%x*W02p;I(+O%;V%XZIa-^Sh^iL`{odh@22C<FhSMrB-Onfy;r_sxC"
    "f~-SKTb!TxUKz*$Gq}$O=_J^IiS0;);BGEu@UnNrMFn;mY7t8;L2F0~cqgtZUr;;19i=e(@-"
    "jc_xQ}~?#=$2Qz!=R6I4L0^+cm%%zO;PcPsJ<TL?Zphb$EYg)qvZ@lo64KdKe00E-"
    "EPHH^WA709^=H(isVA(VfukJaW%hBUfy?2xfMGswE~(3qxqLv8hNoZdyoEG=8lVTjc*HV{SQ5`j!Foz&w8#CBCBDAX19T1y$d${T"
    "UP%V6G15)x>&)5*lnbC35{^x>VZIhX;P;37OJs;s$_~Bd<YqhU;usy`Ht<kWB~%`CkzaB<xr1gQ3q(KawYj5~>rx4GC#x%UD3kyv"
    "xKFdJrZ-ny?ix3Ou!du9<ik)@W2tOocJCVlhQ|jUe-CcvQg+%W1^G%bc=_(fX<pa3m5jsW9M7R2ExmSdW9zTxn+H-"
    "*TCy;k;TEAN@Uj1oL4}`n2w%A$xdI80Tb8M+0A2azx&7d9Ty<qH7wjbOJmObY*<FUY>E1m0-yPEeH_CT?P|z9L=AIweH7(8u6-"
    "xxR54^Q=r)pKJFo&#wuOw93nWDN3%q}nT}q!Q3Tvfjwu+I*a^COXg$O?g9exPc2dSqx%qGBojAROeM&(>M9nzmfllv+X7bv@4rLY"
    "!ZyZpv@Pfzt$tO=<6$ZkY&yznz)^TZy&pUSk)3}V?S8`qyGDxk<+y!+6NIM%s9OOUkKG<@~Ifdnji47@@4y8MfZM+G_50SA87Bl_"
    "FXrEH3Jnrf6(O(MT%8~%&8`nLEp;lPQ#Z3d@-Q=fbAt2}lfnTtti-&_OXTW+#kJD^2)u{W4(?Hbu4x7q2!7Rs-lEVmc-RV({w?YG"
    "^4P_2!qYw$GP#nrJp&25?+Zf*n<T|*dl^6gkK%7OG2DX&MVfGAfTiCKqSfqpl1%su#ijm*N`#=dDx7l9n^c(M!rwmG1cL)>*uoA%"
    "}kIg*MOryJ;{BRH1z$PZtB}zk(cK;@77Vu=o8cxg@x=>T(gBr)+_t3H_xUSta07_rrJ)Bl4x@;>$`31N2obq`^ApmT?IGMB3Blrs"
    "IYyd7v9?LEjXR0nH&Yt6DHuo|u9J1*O{4Te490BkGf|p<p$oSv_lUsrk+CtE;CoS*HYMWWiA>Tlznbs#Mn=zMzv6b*HZo^GmbpSC"
    "+(F~4DHiaZ2_u6!bp?TrQ-"
    "k)o!R0?n=xQ4JrYDf@)g{lgE_<HBh^*K~;(*_}i(1D>K$gN?0rr@#^!hd4aB(<CQqFg>m%e{p4O9q3m{%-jeQTiV-"
    "%dMnDI!_(KG~^lPgqkBVd{Y=TG5~!7C^-"
    "G(Y8eXIALH30MUT~|%r9tWkOJMbwrhag2M`0QRZ@+&^jJ(6n1Q)2U@R&Z9gH}Vqd>}G`i8Y*FJo(Y-"
    "KalA7~uq$l2CR}ta8`_9oP0FEDA*mOkN$JH?91q<tMhW=y1W&p@@I*^C?eD2B9=Ixtyw<<?zdbg5)V~0aqCAA-Fg8a=<aEo-"
    "ec%XJRh-VLt)z`)?=*!~`oCfeKKZi#G}c#$27yhQVyQnqP5+k(<K6%F3M7f;=%Af{`~f(hh4ks1%zS=iED@o3^=?x5iNyCiI`#W8"
    "Er9ATlvS7+jW3%+WxxqV!sB$?+0mz&VsSZq>XIp%$`@J$^>b^vNu8z;$b3i0xxIS%x+77QgjAdD*gC0S>7>SInRAPM)&uYrhf2(*"
    "_psrt2bch=_!0y;G7n9yayG;)#F11s&LlFOYdAr?=F6I3xfzLYAy7h8>gkFiDD-"
    "R1JDRran<x#s7cIuf^q=x*G;ZdgL6g1uKGBj#ds7K)9UuE>VjKg%LD?&~R<*_m_Ox?Do_fbR&qUk<<~<0(UpRT0$n%2Np+-"
    "T0OAf93qc6ZZ@Adydbj3TW{HM$qq$T<<7S9Km}PEUOhXch#*a60&F+79s{&@ngOuU4*SwvB9M=bmkJ1MNlwuX3uuuPBhxYFP%@Q("
    "{pujL#Pz2{1+Gavoyk1pf4O=O>lXNHs0zRDl^RpNNvJG!ogK#TC+IRWJe5reo3vMq*-lww3k?yD2O-"
    "EHLc=o9#B|uHiDt?dn+qqWixREF<HoO=lViknf7)TenLzB|EoUH`^tca$Yy%7EPZ=kRBK&IY6)T8{-"
    "6VKxlsfgJM&3*>r!S7AE}vE{L3RDy874+5dh*xpqsQcoOUD~)wRHe4KI@FI1Sy>V(?~|HWx#T$kTk5-G{+lBxYj%UL?^zKxk-"
    "5j#l=<d%D$tv|L|Kn!=sT_nRbaG^pQ;4u9b{vZINLJ7P#&HR1;V+a@AfvWF)$AR+@az%&7mg`352}j%%%)U-"
    "*q14J(iD59>BH9YjiV{MhkU6ef9Od*%EgsTtZg8jIQ`H!#k=f9Fq>`id#N3&|r2*z%xw+KQ>+-"
    "r1(@a){^=x;po6Svl1VQ|Jc)&k}(y;Xq?(-"
    "E8xPK{uOAGCfzdz!U@~kVGi?BJo^=o&giOAiORoY7iGL+Ag3SQ@+VZkkF$77CV?^E9(h9qh$FEZh7aYFA_!I;ikd&bAE#Wf*B4oR"
    "P-tawkZlQvXcz1*E9ea8ZZI?c$mr0O_Pj=6%Om?aTo8NexHf1e<q~gCVx<dj;lC?v<gh!SXpKI8?d?>a*)kah&5AvzGI1=7XnRWR"
    "iErv-fumk#sQd^+-Dvb98ye+;;e{+Xp#aWH(<`I9R;-"
    "wb+E=!HVfc(vxw(cUd4o@@AG0l49lU+NYlT(yNa32(psc#djRmn!KqOvmtz@dd#pIg!@cYxiSIEiVOCHY*B;tia>!ReAw?KL4#g?"
    "Nkbjo58x&jSq8K21I4JhDu07O2m%A*V#mt_{krhHeb71Uu#I5{Fqa)~iiqA1+Wmc`5zzAI0R=J)Z)?CGPVo^hS6*b%yNhxRN==cw"
    "NmZQpR(C^AGgAE=_`)*}#y=y-b(i-"
    "3m9*!BtyvoJ68aCFC(k&BqwvI_)ldr5XPzAH5QJA%8_h|zfB$yfvMnY%)2(sQSbdb(eH7<6x+YJe{g-Lw_A!)tpQUJqwM;sEra-"
    "Q*3@}Nu0#Vs-"
    "&VXY>;viqJbOAZY**WI;c7s4Q!;c4ee39m8J1B2(%xWvJg91;iJsq++ChImn(ln<}jEPVAQaPP2_u=zG7CrwObbkXKkivPyC36%r"
    "Cp*jO_k?U%z4sI$ZIQVIER?&vXCIdQ*ZTB`{LeIEq7xpWllSDiz``G3=U$#sLbEWnaRrm*U%E278vA0*Qk$}En!k8HM@=<wOaYt$"
    "3@jE;xoB%i(9#lcRB0=+NK|i*aco5i_MI&PWqg0%S{z7u}EQonoSe$Uyezvt+FbBg_I!%u>5A6;(!b74zO3;I0G$S@`;<M!`9ro5"
    "PV?4O>_)P7Hz&xAm>>3hOPO5r#kd9g<!F!NKk-uOa)*5p3?xL=}@}80D8!wibjVT82YIQH7{9)r!*6)JqLk<77*~eGBY<@%h-"
    "Q9mK;ea06m!XZP$&rS^0BaVQv?@W~7mq~bx$^pu6Ci_!rM)H<?=W!T5iu*uvnDStRwq(pR*8t|t$<3^`Nb#WjG``DY%49<H<Dsy3"
    "VA{}vc}s-UUnL+{L_kl3KDmQtWV9JzKppE!J3Z$mK|X1&c-"
    "ozh9pS=3{C#OP$E9t718WSJ28O^vs^}(IW_+qVh0*ihn)}7qfJR3hG7K&{DuWep)L4VXbg=!nLOr?$$X6vj$3j`yP!Bgr)Hk>ceD"
    "wmeU`=1gWCrQ3NMTnB<0)XKM$6&Y-t&`N(1U_?cTh#xPfMu-gYuedA56nYl<dp)CQ=gzz1~|lC9SsBhhU_Q<ze16D+aik^jc&wG)"
    "+5HMV?+`r+-"
    "l@grF98IeyECkaX%tE(O)bT_D%*#)~h9z?e75QZZIH&?#t#kRo1lWl@95No)My8|*AL=3IK+&6Vp$IsU|Q72jnWK{dL;vY-yCjV{"
    "Ftz)ZWEM#UA#+u%V{A2`lR~+>$6&TsDR_(Id^x9^gbjZLRZ?C46H^mqQsgd&zyQq3ao$TZhil1^}=$MwXjvM+*?R=m+OM={}Asgo"
    "o2QJj{t*R`Do>9NzRV?Fn<Jj_YM#T(^_|&E<l8Jn2Fyzb0t>%>swAglDojrvTwTmo}f1+}>?Q&)5?Z4!)3KD4A{=+=eaYJj7CTb("
    "rcUFDL5^7>Q(D+P2l7t!H#)y1-=?B;W4=aG7Moyae8(S8^8c5&b3*xUr|H>KlUNyeikhWh&%9iZF)|(6-i0@yuI1IAo^Ix7k$Vp-"
    "BkV615y-Q3+(0p*7stNNr@<K2tiJ$YQOWqT&6%*8y2I6@&1ud`*-"
    "MS~bwP*pRSl5XL$sq<}2fH9a+E*Dyr1peckW>gU)Og+i+O}r{w83{=3zcngb+nQQ4IZl=ZrN0Ut~D(f*=0_H`L`-"
    "SY`Avmy<n!&&C2PyB;BMRJRzW!iwN(ZI+YKvCieF2qqBaJf1~Qovrf4Av<Z8%j?y@k@V+6LCw$)$$*Z45LXuwUW%sR09sM1d&N1P"
    "Nca4WHTL7zE-)U|MA>x(57Fu_gdD5Ki{stv!Y)|Wr@FXWwS{I|89CRj$m=7j|_zicu-IPUfQ+d?L5LXc@r!)#!VKP+o?-"
    "3IXfKdMZw#jDD-@(VVG`R#OQ}mQ3x&CzP11mTeKyyHxfxfhcr%{s8g-"
    "6X&8n6*zOX;7ZW;TnRgkA8^svW6PiUJqPp({B=d>=M|F0=y1xp=uu9|^w2ZpAqZ=%S_ZqRAvHca)-"
    "qBoY|bf_244Y+8_BMKTosCl3wgl-^QJUex%T^~5<-q>N_27&L0nvY4$Q=#T+-"
    "0ov@5pJHjlJM}P2ms*&B*1#npOZz&oSV5BU+kFQHD*Npkg>o{6GXBs`k9?FY-r&A3(biKhJTxX|9S2nOWR1Z5gq%)re=e-Vy#-"
    "*GVEDK=VcjZYGZe_n-Y(4A*|;s}TXsA`50FMN?PuYwvH68PhiFO6yj1-"
    "SHA$b59w!TM4mnBLCZVYkB>{ui8dyZ<f&v#`<!PCjY>aRxq*9;W4^1+I)+A<(iJYu`0HR|DkBOX3GvXnCYXk3)HHbj)ei@AVEaJl"
    "p)Cb&H6Wyr_UW+qmIBU_)fo{j%^HI2lHh%$0Q&jJaf=n>DT3eP2zeWMOF#rQ}YO|c^27bpX<?vE4sJWRy$_Jk}ZiOq51^a}kcx-K"
    "qlN9ZVkk8+dr>bHa3Ic6_7N*DEawEhN^;U8Cw9<KNXi&>xo}6kP8=PpjZSNF^L`zUpikjVhmvua@l<{?HW}%p<C8jvkdGsteRa)+"
    "!$H|nbdzZU$nXW9Mlf-"
    "c3SwBM%Ny^3iifd8kBh*}=En8vd4}10D<An*d+%>ZCX3e~86tz_A{C+#r7LJj3fI#EtQ38qXx(kzAiTqJqqgr{;nRAXU)i8zzrX?"
    "Qj?^H{-N`pwZBZ5VvuqZ=ck})6V6H)znlM)~)!ce9_-qF~2ihu+d`<64-"
    "r}eAw@NO!p+|N9i#`5%KT$<!H5c4MIM%p4z#eirL7{YgBZkE@IiRsV+&-"
    "s<hXX{!>AUBo=dH%2KAK253bu3f8IEOYv0|5sMYqFbuX&(mYNPeWqwWw3}*EsJZmq&0~u=53DMzx9oLakwD-"
    "XVoFqV(J&fCitz3FY`*Rp>!rc-rs4`a5feu2yWb_UmciRUXjP!flyzo*!7GwLRpOmn?W792Rk_>Sc&s+(_`AC(oMS-"
    "W>e&Kbzz@E~7S+#{ha?;|IITNClH7E@38#H>9&1a&7(DU}}#xBTClVSgw!j0dpx4N=zj{hcM`rKF!7UPhN|wXM87lpBWHFge!T%e"
    ")YMT#7id&YsFhdVF1AKR%D+wwnM{knF?$yS@v&@il<zk)sm5!kPr>cvR5^sTNQa%jazoQT75t=#enSf^D_pUGKv)c#U^`5dmyAqN"
    "@oov&2pSnCOk+4YnMDsTY6Jg5dYQ*1G;Fgz`D>fRpg}Mze(PnRy}~(HhHrY7X3+d)mUaCimX~&n$iWh$(THa;WYf3y)i28?vxpc;"
    "u6ZNt;A$9rl*KS@cYWE1cU&aV;pHd%M3sH2SnZ_4cKFb?uIQKlmOlvLa9Ru4Vch=900!Zq2)PW4lO9!fdJ*b*Q*SpH0Du*_M{=Lk"
    "^%}bmfP62o$a5J_``v1Wv8x>F?GRcM-!2h?+_ExuiG6IK_u>hd^3AC_7tQNqkrYel>M3(Nx<Yb(1-"
    "GzLW2F4wPGA8S{H(QBxBCJG3fIUCJBnQZ8jdgT6Kam%r47ERb<rOP%Vqdk3A@;&nK5Q8@@g5;A^7fC*_lmluSLz1YWJ~0AtRly-"
    "yi?$=H;Q4YJsENMk?h%ts<`bx#Q5h3%HE88*(wCsEG%DEF+RmQEX+J8;@nnrC+PY}uL+1eYycb6@q)R_9j_X-PH=JC_A&1_o(JFC"
    "a45o)=AAsjb=YiBo4S7G*4iUK>=tR+B*r%(YM$50_pWHaPCM4P@VZgw~~+hr&%KnWtPkd|16UG+ju;Dbee(yrSDG9+y_A(#~rSg?"
    "#|o7vHyB=)BUe%SDcG2R$S4hIvp<DpKrG%r}5W(HS_K)u>@C@<5+g4YOG4%N9nNHI-Q&>>Y3fTHA}5*X<)-"
    "5I};$(enJu7l4KVbMzl428JglUyvc5_v3u{RbIxv99QH1csD^`7Qf&99Ikd)q8|-vGB-ebB4#IB+yy880r5k%&L~SUnIv8sd?>3G"
    "pNUa`$V;X}uJ|B2h%BoG6n2P@@G#)$oRo#T5r_4pEimvh?D5A^1DWQk-p-"
    "*{8r$K5^DOs=ex%(x@bYglz;hu8yX}rhP}vG%Qx53d)5powt0aEle?ZAtgMMw6o+2c@!arcLJt#A+AW6>h)l^PYxAyaelhh207%2"
    "zygOX;a_HOa*O_f9xcQ7*5(mbncho(%MoLhCRhPj`X4S`Ggv?1tW87If1$Y=;#j*R)AJl|@|u@)K}3_SKNUbrefrMu7M6V-"
    "bNC*^Er>OCB&j_nBB<!a)=I+BAkJIO3-O?DArMaqtFRzDTFF8+qdeXOfo-exY@)Wca3bH+r0pO_;DTQ}tmtP>6i5L=k|>?U;+&Ol"
    "ih#wPAb>k6U}IhK|u4K%O^W%$B5{;gm@7r5mG3cz$~aUix349M_9^^IUoYm@6)VqX!nx$~ge-VDv~m>7tkGN&o$g!*qUZep6`=3q"
    "naLTg9?S41@V5U{fw=G3dVEqb6F=H5>hsr{%2%?mRfw@WR}rgM1<E8{)SXOfT*P%oQh29>=1ChgVi#Zsg&X9`b?Vg)@;5tG-"
    "?LLnP%f8t}pKbqvbY{PM`<%$qUdo7w?a8tpdzcL$z;C7mvg{W6kR6{!mK@A9UFV|K5s4vn)yq4Qx+&pD|$Veb{4o?GNcCTZ)o*ZG"
    "UyphULfmKlt$fb$I$`V69)mH<9F0o22C7)R3A9?*X??8MMD6rhMG34hwt9lVb&oUH}*13zf&K40je$a@aB`}3Jj;J7;Xu{=qg3Q0"
    ">0Pz0c^r3>6Mn{GOM<M9d=4e!=jGDTZe7O>(w`IwNwherae<<dferDW>&od)%U@erQC@FXOEoG%NcE7E7Pac>&a#RYHa@gPv+z)x"
    "UoM)bz2kntR<)%krA$u8YJa&Q6X(+*1g`vM~vN703Z_3q8V>Sqc11c95WO1o-"
    "{H8~><*gNY)C#Jd>nvc$;Ih1b{s58(lD@CEwxI9jgt2pj%Tst3f`2(>>P4+uj?_`(G8adtWZPBDP(P_}emzO<jAc_5MfDHYM9M?p"
    "eXOZAFR80crr95qG{-D4R#GrCgLdbkr1o=vNidkNIOr-OiSx)UhIg$4^wR?ntC0*Bzfo%>z-i~Ru9tu*8?_?}kDnA-"
    "i+NI)Vy2go8RZ?}`HZU2nrtzd+ylgT{WTRF*NEWUidDiBfzk2A$n(yMsfOy)HWDLIV#g7*ne_LB4;WEl_ygL|JZnD?KtJ8e&x#-"
    "{RxsM!FipFSy5d(c?T~Ewngq=zFIcBbE-"
    "Z;9^XSD4LNduZg$yMqsqygnM6)T&4*XOA=TTgHe+EM$IYO;nYsKgh@J>4nw6@9UU6jKn2bTLIMn5<Nbl^CoWwIsXeCe>j95uC3TH"
    "tWy8N>fwlt8wfpzj7SOYCk)Ez2@R{+_viunFY$Dlf9^mxv`@pA#Zd5hLUhM%3xvyvCttfFjGWV~3Sb3;r|#Kx#G%CFGe7dBStgYi"
    "&G*5XPccdM|TNAb(>1zO=%6S3!dBJW~JHJsL*J`81%}_qXeXAq!yJrzCGbU~p(m?U9%uc}~N{tG(N7?Yk8YwC)DZ11!N;fa~jd<z"
    "OS6T5$|akA_Ru7d`o~bZx{&v6b!g*aYfKlga#R(8u#~0c)FW_{HagVDYTkcg+j~2e%FL#dxd20|}YsfWZ8S9rm4*S;|(`6ij;LbL"
    "ZioH19xmkAYb!>Db*duO!c>IP;9j0BXKez9q1Fn52v7iKFeWwOOjVkla*}B(E6;Dbutd@4se1&s0vE+09O8iQRX#0VL&cSWe1V7W"
    ")#|o~K#_;*=<JIO!w+M32exn>aqZmi2bYttl|C*H9pEV>czcXV7n!2NaHFVrVx7;vX8OrkfvJJ5<>~%N{Rg&3w*!O3cfd+h|PD4d"
    "zoL#Se!8Z8;R)q4XnaXn*Hr$c`b(iSHKAgM@^j?g<xa)+G`lBD{zzQU1$vJiJ=qwlE$oF1bWfFhDu9j+I;Tury|#);X{PrE3L9E9"
    "XsdMY$;0alif3IOFf5`84Db(?^g_W$${uX%MU?)uWxL?3O9BH6FsEPUf8Jf!xJYmsLecVX0Txm<DmNm)jjmJkq!{=%XZ2NIEYHB9"
    "_9J={BKFa$oEbQ$Ej`fmj35)2~-Lp+kW%gwqKaTrF*t3lPdS*c;>LQvcGLB#lxmmJ>pBN;{QJ=~-"
    "z&tQ8Vggvx_ml9Xn63f4LGHl)a}ydp3&kQKAeSWN<;7JpL2*VmSRDsT*qK}q7lS8?#$8U=XG4FQOWJ|iRxB-"
    "l8P;+2DW=k?BB9MB}4q``OBBcyd9KgYM_WQEC-WpOCpWLfvaBs-F)f->tjW814mHBSyrsSJlHc3@iYr?2Qo7!AQld3A3=0Km0a-"
    "A5SL7?4ei5IjMAcCB4HQ+xkV!Zv$4G3$}>Vw{_2R7uQf9MiRBs~iJq1woZ*UnW$6-"
    "<C@F>Bw7OF}w+#IJq$Wpq|P)q35nhG!OUgTGCF15~mIFh3&DBl&k!(+lwdja`(Z#bK9Z^mLhX{8OPdI6Ff*Q6|VTl#sHsoAmM&VR"
    "4bV!fKp=<H*4Ws&mZX~DZbGG@7hx|<yFrVdKPTBW&M9;R<~Dy&`2P0@L*2xe&fh17lNZZJ3X*M+`MpR<<r^`heg+TqKLuVU#+H6i"
    "-z8uHp(QP&Qdk#h4kp7=(&LJQ>yPnPA<DanKI)b67k^p>n~g$00rRJ{H$CWi%X8k+3nfub&4S^_sP`wBX37=nY~O7;IC|ki5eIsR"
    "=5CkIdXCXkm%k;hNl^r0oD6wsjim310wF_Rd|r#pEfs3^HYR%_IpNv?+W2bv4D%haz`K#uOVQE#?Wru3oR<k9@tt9$Lh|O%lX@Pi"
    "!c>8!B0z!inC4+LpIa4x!<p6{oki?Z8IOLWhEoz9C*lkM1lIC_g#EEO@qVMsOik>7L9s6G*I$H1WK;9LE-"
    "b09QN7=XGlpvFEX|H#Dh!XQu=t{GX7gKpxztiVAl##pW5UMJuOvmKY=QE%e`dCw8T(1kZ(gNTWvm80vGes4?hccu{hJM4Bn9?Mw$"
    "0w3Uoy~s2LVRb4hdlj)t!lh)h!u`ETzCNxIU|MjwcT#wkBZMDbf%*NmG6njm=`o_%ZEz>2;>O9vq+c;F16fs?~xU6%W59-"
    "J&ml`o!GY~fLdvb;|jmi(-2Ay1-"
    "^#h%m&7ADJ*ne6&X$3jETYEb&iRtk~MaBvAy#~0E?JUe26)3>%F)%>4^8+x2(&Nes(4rR2()~8u+kZBsXhqgG`*a0TP&S+dX0mGW"
    "36;ISqy$lKHr@TJsq^2M0tmblu5p(&XVYaa0(m%sV%$APK$+0c?`H<m`%m|M7mmC=37a-"
    "t|*}TrLT=b!9Lzi&14FJd6hqTK&XrnrvVu9>}7vn0Y-<P{W>C|0)V6sj-zlC^Tn{|UKY;BqCzrEUr5dqm14DYQH6?Cn9V`b*?Qgx"
    "=WjJcs`?Pj+z7PJp>D@fByv08h#w0$Ukz2#E0JNSEv+T=PtH)z2UyfuQ5dV@TLa>#tkgWU4}XT*rBfM(RFcc$Sa&9vE`c*JKZoPi"
    ")jnN3S%J76p*sEDq-"
    "#mXEVzRaXxfR(!bTwT0)Kw$ia=jp<jYji#MTDdCId<l%DlXK;d)gj|$Hq990d590NM<UtOPHn9O1qXbv$+NbDR)bidN&hfskw3+v"
    "P0Hyxs>p{xOerqk;_}g8{o6S*sWPO<=n2)(kr(7=BO>A{91%tg%SL@5#~?`BgKm>dbcU+o;41t~`KDSbRQ0WR%{cHvLm))&UW-"
    "pz3dIb9x)R67JX54@<Rko$A_`&CEJElv9gJC2`D6n^%h(cPhZO-"
    "bLs5%uOI}T_ib6kn7^%2^!02DQH(|*o$qYv=_Cw-F?p|lfnxA<fKIf*jEOGdFz-"
    "XvlPT_SmU>!j=F1rIv{v@zLn}n@}uO~A;G{_RDT57&?#*d)3cA!Vpm0Xo2DCr2TxPm^D0U`ECtNH-m_NJ*}y+TRhBGxd#$33l?Qi"
    "OHRg(O5fI&y-"
    "6qYUIyIFZa0?ZwYPk6{%cQ}Z56#!w8_KIqp~pMl{tu_w!zOK*pU4BM0O*|t&;>`I$!^Sv;)QE<J+yKBV{Thd3a!LETi`*r~XSqb>"
    "c|A3AGhHBTd_a*&g$cD00j_JQ;Ei&%Ypc8vHdxOxKP$*%a>s5JZ-"
    "Fr`ConMa%z^Ov~(PBIcmOvJz{Q;n>)lfh>U~N;Ly*3QN(m|??XSXFG-zZM;3|nQ2%EA~%TXI`2)ew3nSw?^&Ai(fsLkufcCgJ`4!"
    "@MHq9dT<lgtu3*Uiv1ROp_Zv70S4P;?3hkMvt(uvL*0+-m836dTG7d@ik=_(Q>~HBWvHPrgbz;c3EFJ5Yt$T(0HaP8BN}-"
    "+7p9MvTYcd@i0+A&vcBpE;hP>JS(53tnWwyh$$uWQ;TY4?29sFAj);>*A`L8Wef+4xeQgN2|%WW@RXbxQo}lS+gn;`kvlr2!1ofA"
    "_+ikw#BVS)fDQ^2=4z3;-"
    "|Pc(fU7<k3c6X>R7B>1NLqX@qB>z#g2x3HByZ2jHKJXMOlmn7vxx!SN}EC)ir9m(gnYD97U5mS8(rx2Dysd{lFK30N7>zMns-tKh"
    "<jT_j0BU_M!Z|74fg0VmP0Sg+Sm-3`%0a7Q(2Yg2ds{$icy-E58->d1qk+EhrsTW2m)Y1-"
    "%NUOQ)#OmGlcd$k7~SVX8sIe4U2SeR_+7fvU9MsoAf6?hbll{jbs48*;z4pI1pfE{fmM^T)!N0<B3W^4R6whwMuJR0}nON&xa-"
    "7SE(K>p;#ZALKdSoZD}fE-cW_XWlXeIf_+xxa%yTB0DL@Q#|voT%WbHim?rclGEMgySTemzWB3uX#Pwg&TeKUtqy$6u_*!7Izgof"
    "vAsNRjM!u*0+>LZZX_eM4l@>Bsw8)z|L^6mw<0a|45WuWgEUX8J$&5w*?IzsH>8y6PdoGCLcM1?|CC0%@SZM`)lBK3>vX{Bw(3)u"
    "X1x-s7l#3&57C=cGus+*gSjOMAWf!s!vJG#XlDrkqIn>0pn-cxJ0u=)La)*(d1bb>oHTx^^Z5NNUYcuEkt@XtPQ6_D;dhRlj9x|U"
    "rb_$E9tfe@tV>v)2f^#(U0I`6m>A$$yL(|oD;Qn)}d%T(5w|HH(%7Cgj&KN3#QKom;9I*d)e-"
    "+i)9S!u(d`)XbN(LhF7mAZ79Yy6{EQ&C4V~<)1SteIxYx^3J!37dyqN`QFgbi<Th2X%}12SW=n~~_lxR*hm6LL#Seq6%jjcmh1vZ"
    "{~}n3-n&NRcdR6LkDrih?NtCDpJdQ|Wk#{!Ipqkp*)m4F!a99|;{*z8g@OK=@sDIlA}{8A?;zOR-"
    "!SJox=CkJH*1d$8d|(?rtnD&bEP@(v3D3Vv<3=HW1fTw_)51$1<0+CcU)^=E?&DEMoIkI;iWxy}Py_=FPHNwQi>@Ao$0Xw4>T)@K"
    "0<0UM4DDzs|m)vG~cVQKKM9Rmc{CISh*h8MTtJBqaxp)Ny^-"
    "2Jpqi^1%l{pOT+!J}@np}&^%(qPh0)G%IddQUqkA&_?u4#}4^={ca!G!JOaDu=EqlSeE^3p)V!XZcP7sB<=nIa=OmC=#X$-"
    "a4PQMEy+rwl+avo?zc<G`KAB27q3z*3L<|LQrLd;&aKgj}aOaMDJT?aRorMrk7>;hT@BzFgG1kasd_=%+g(LuYJWUeY1uRLWhl7s"
    "3HMvwb>Zww$eL-qJdCjDlmTg<JJ2ECcP>~(x>85Epq<LRtu1W$3qE{B7#r4)yHDCk&GM_gh)~u_cA6iN&BMUJ=%v8U;KyYT%7x18"
    "Zis=-T-XT5UT=`wE|TEy%~vV8PMm4E_U0&hn}=KD)MZZ<%|6-"
    "9dQjWy6ngxTKQ$Lh#czFZc00+Ed4oyxblzv1Awk3mmujZHT3MGqy389K=b%)Y%8}foPM`f@U)`bZ@J`d)GnSvu@@NL8Tc+Duwu!f"
    "bej?{T}3FJd)NYU!z^#9q3;a_EGu$%&6G@t8MXS%3!hhlB)=tKNj||_X(TD6%kIYu5?I(rFLb3<$xtk#q_ld$HoaCUOmu0KUbo^i"
    "w;FEM%uTt;Z*2T5uz1QgFQU+z5d?`SL(F!-k+3fT{|Kte)LNcun2g;&)o0=_V)m!)=-"
    "$ap1H?JRTcc1=oPdS^OuX>g=UobX5ZXc}+v`t;i5BysLpn8W!OMZ#Q2=NExm#g}fTm(DE6bRrYE>%g-"
    "z@#LbCYuAaKu*rJ+K%mF~%cb9U^RBnx^?07t|X1Oy^C^I?o6(hQpx*r03@f18k4|HimFTUS@9bw2GS1!>%oj8a{nYi@KM8vI>itA"
    "$f_1;T@Vw$_FqIFzG~9VJ{&T)}F_5+y`^lmM|RS$2Np2YgttU4EVeh3xS4KBVKFy#ECl^tM^l&{a|OxMewhQg742mvx~DA=2z3c%"
    "LusT^+p4+9HodG!1G7q0WTkCO?;CB<<yI>Q9w_6xeUUNnFoz*iv8-*w6Ua-"
    "P18~4HCv{2<qHD%A{yWwrJ;qzqs1^RGx$pGw{vX>iv_iuCez!%)Tu#ymU(08dk`tWh#J$Mih3s+J9vgb>$iB>(5JPxKuv@-"
    "s62rw<1mbf+~kAC4S)xnoJ<BCEV?HXam6xtchREpL-}k`&hKp6ox$Ge^3z2Dm>XVbOY$&j)43{7RcaLzn8|=z;2Sf&KnFAup&Lz-"
    "m5DIu7!b1gqSqQ2bo>bFw%%(M^3WLQr?>0}O;;prU@^)7{nfT8geNd^n@qV@sp3hI0Bw``Wx|khw1Ue@CSRUpNT~$lI=T%+bk-"
    "|v`e%qeTf18`H)BWow1|5>yDGEIT?ALi^j@jRmiJtCby}H<=W02?J?(8OyQil`VxZuO=-"
    "cTYXKERl6r6zTAA7*4D7u|obHNeokWU#Qp-iJshnH*2P82FRSdn3Gk<SHIOkH>?PIDcP9F!2m_+Ja|g*1W9^Zh~XLkt<2qcAG_!>"
    "<v54}P0Ao22PKqT3;RvROuolLiruW*0NF+6HDev<^oSgJEm2D2^x!v_`kU;HJe}K)4JVGugoi(|UPU#M>Pkjzbq{mpE7U>aa|hOM"
    "_-H|I0cqehUOL7=`hJ){ol(ltodF=X&;sh>Y3Xg&Ta9J}}!r0bfdXvog#@HAjf{wUNz`$12@1_L2q`acE{!Bziv_ffKvZf-"
    "9=f_vgB4=;cI$rDnABoGp!hup5Mj;$MUmLvw5%%Wy)5@%C6|uwN~PK6;Q0pCL16!~*LXyaqb<k{M!QN)JM5I$W({v;&iwds~idOB"
    "+j>4pMKXgy3k02&444J$Xx@74RO|RJFD7CCNBYwE0UIiJ1(;d|0aeacvzW#RG_&lq|kY=uEV$4hroln`c0H(kwfZmN7t@ZZHrQ>="
    "k)S&blC%$b*i9KW?G-@9<sScq+Q1T~f@_Ttqyv+vHVZ{vRH{l*jQbn{q;JiT$K-"
    "TYoNO*i%k*ET3SFjps}Qn+VM~1=wm>jWZqIMy$6?q6#Ss(mczQr>8$6lQ1&VBBEL#{s~3~K?c?IA~+QR;NM`N+rkb_MZTt}*sU^$"
    "ZpMmdrUb24KG0=6wdFk(i+qmnNfX5m%OXY$6_9LQ+2-"
    "Vv)^TG+SuFXg+0Kh~Z8Gx%8uf+|=BbpNNcHo|Kyt;~SnT*(tW9}EMoryAZjO!M$kj*k_}N&9b>o<ckn!kt6ZJtE!LV@V&eSwqJ6a"
    "nl;~Z}%73jpHCC`&R|8&Gqf6kHr`MR)>6)(0fhxD3zBPWJMw6G%2_X?T^XBNQX6nk`um#LkSr=%;GRimkXsOWofh_tmx#9Q1is{d"
    "t2B?JUMR_&072ecKZ$n<t7Yhtb`b9_bGp~lAjorw0lGvelQD%B0$KsVUvmg^3~lCMTB@WUB+Nb<C+h=tJy<a$r(v|zF??H7Sf<Wq"
    "j7&BcaA>xEp39K1cGZbM9*sfku*h@KdCzvf9S#*v8~6ga!bL!8pGUIcf0mpT(MDL!{#Uf!cgH2UPAxK6p3%7k+za#RfFg${dfF*v"
    "r>dxzZ3(h1zI-u>-wgMLK0%jfZ0fzGs|3#@_1>~;@;v-"
    "8T>RC1@WORQ#@jmd_3dZld_d<B=5?N6TY!XDsh9tcN8_ov2}+?Y{zaivx&aWFAMBjsq<(?5f(B_?E!Z<dsTcdPu!Oa1`f#TmvxCo"
    "|3x$xTu`Sd&I^wAVUK#h^l!bgi*E1aO<Z1-"
    "8{@bVOMUy~jV4VA`}$0Xw+;15+2WrAY%11>CE~7OY>WyPy1y%dwYslUC#YNVQ_{9Z~xM$jZ_R>XjVJna<d69y=)p9aG{v?<_!O(`"
    "AA3+p$JR4qsqN#A_oAx)ZGe_|5<M;&HU74K<i@g}=qEWXlCGaE_@cJk!Wr_(g_22Gf8U1U0vtLu5Gg#B0>v03k+C5|iFbW-"
    "J7E<znd{qX5kuI06=jsI&h>Qa*_k_Rzk-v`iw<vv8x!Y3o;wG3D1jFf3PUB>-(eQ^Q09kdd4=6*C6n-"
    "u9drb{%?S+COg{2IrP{B%!5<u0W7XUIri6ur4IWmq5y^)($LjXr~JPhki)xUSfL<f^v%TOuCRLNPCmc)^?isfMJCJMw1Kx1T~FKy"
    "7=+$ksXvY0qo1gv$K0#4(|%9RyqrVpkNJnv1x9y%f2snlx@v8z;H8+PvQfs{AA?wv)n?)MeBzN+^Rt$#w)m!eHvn=qyoDM0Db(PW"
    "-;H%v6??!s8MO0%)%7D6U}dJupu090n!oCv&m1o@G0{ow2ht`bcxV8EmX3u<;bWY8lVA5ccbjqo@D+{e;;(?a6K?7?#o_5<a>g(3"
    "@d+&S`n&67cTGVVpZcRg`1048x+k3r5a1@7!{h`!P>RdS&UjpA+S{3etzN0jEbESzZb>jvEkRzddiDLRHTI3!_5*!rl*v|+31O0$"
    "PHzp=I4V*A5}L~;aS!hv;uIar2qDyeT(Ia&K2_btbGb_fMADlqrCu9t5^ZOagJ*K!^^YD)_&%S2KHG~fKW2xnDZkU^sIsrXKlY5y"
    "l>g~MT^{itD=|~H90py-"
    "82HawE3ZHVLiasZzKWkw3W}dRg9wUu|wsQ={c;D4OJWCP`WGYvb5Z_O9Bq!^QVk}<+>DYxGmJi1nqG;64EwTM~KQ$@ZKEKRn*dg("
    "MEC3VA2z5=s}NuABC(LTSV*wqYTc<k9dkB;#EBm-"
    "+86C3tUbVDp*`GhVw(^5)&epDC_=A8iO)AfLB&DFbZq}e%HnOGn}#8i6aK8Umc8SX&eF+O<&1~Fy`XYM`qh!`VU=6ai6B!!+R;Zz"
    "#wQq<9h4>unSny#0&!%&!upKig9{jSZ&nM_M#;sjI)FG(+Etv^6)RJoNP{I-0&6ell?lRqAhWQYX|rsY-"
    "~z);Ge}|TvgIzd@gSm#L*v=hvwN1oR?CB-b=(<J4XWHp+-EK{-"
    "0ck1K*o<%v=VunlUH`sns^*n>u(U@ge<vqyj5BCz8@B#XTY)@1njijitXj&m>fEWS9|zEuh|27>xGtGIEAaTV)hkx8_mB=Yt2M=X"
    "k_GE)_mLOfY~L;Fc6qY=!-&g+%r@+Gr{7x9-"
    "u5q!yTRAcmb@uuU0(ofvG{c)r617Ldh|P(nyHWfjFR*J5Bd?Fh{{Ox#+^ZSuX#?<QwL%0kkD;{_sbfJVlz)6+cAohkVrMdWgKC1o"
    "sBDnS*AJWv8*tcHf9*Yp{t6Gw@y?4F4sd%~Fo&*%Qx)CC?ZBt;M-3_qqjO{f~|;Rc$heOcJuexgemM+D-Q-Nm+2fwzuR0kAKA3`W"
    "j@URn0as40#_eQr3t&|^sdKi7B++~PrY1>pqV-c3-"
    "3eaSc?m>(dhZ%fbHYdIE#x(i@)#NV!AsR|7A`ACsgidyh<!LQ=eSj~;9%2Q|q-Q~!;=_ogZxE9$TcrGf?O^J{V(-"
    "yUN{R9AskI?X-"
    "!A547iIN#BByO01_?#M3@am`B5S)@@CMWPbFD>aSbWgHq@YSt{U*ksYIsRmxIK5<ywTp=T8o(_kBDJBq^Trr4+vkAF^rZS07ylg6"
    "a5pgD-S*LIGF0Nwk*<&DP~6ZsG#HqH*VhJO;96*l$7`S@m8YGg4lwr!g3?Iyf9;(%<B-"
    "oMy#&jijiW`ijWPh&d`K<bK;_Vm5)cC@QFh!}TD~^p1-"
    ";_A9rnc`2#+TTO~vbUyY~JNfSVV+Z8Ulc@#HDvN3v%C5+F&=+O)&m1_%rJiM;|-8-"
    "pqLD@okcKn+L*B6!A{O^7C34LQhWWBo~k1yhLyfpgX0)UKO__G@GWK>LLhl^X8&W1}WxD6AB^E3;yl1YwWa%qtbn&}2eutI(t8IR"
    "9uh#n1r%mfxtHJTya+D%JKLU{dgb8Cb33CKeXIrm0jN&DL7bGvx7Kbbs*grv?jC*R$qt3ljdgmyKXpi#ca$c{bG?!Y$7TnVw+HBu"
    "Qay_Rvs;m0Y*d2BKsHUkpgSAPXZi0W4Q;TGr%8bd%s&Yx8;;ClFX=Xv>z1WB*jf#&U0krZl!47TJjmL4knywh3^oA=e^fk;96>Mu"
    "(&k^7*yD5SUrW%ItUe>!dLclj`xO6i|pEu?99#F@~tTAhyXY-"
    "4$M!&gsd9PR^QT)r}_l;7WvqDH|_%1XMChoVNnNU+$8CGvv1>b3&*Ypw=}q{cJOiv1DB9-"
    "$46`Czp!r6!Ber52tt>DxS<R)NUj^SP`Y#wU!8iIRP5q{N3V4fT)RUX%afNNFYXd$z7ux(L!~(P~3t+7?xe_H*ebd;(rX=K;$oI|"
    "I-"
    "#01IKCh#SB5~=D#zPg$IuE)m~97<q)S#s+<fDTSPg;u~K&Yti+&mXmfR%pVA^S<`XNox$MB}<Tj;b+z^b@PRNPaLNJBDH4#o5FOB"
    "lrZ<kS*GfG4e_yQFRl&o-w{VKiC%OD+w!?XZ{$}bW9G>J{BXzQ^NHLx>?eTQY7#S!0%+-"
    "!L{wqRpgKc_aVXC`6AgzRVOC9m!LrhG`TS*A|Nj1Jom1%6i#>HUxly@-BSuG$G2uxv|HhsWs62MuA-"
    ")b^91Rk3CMh7t0szJ1ynYC{UZw~x)<Zn8;RZ1ieOW(!_qW!aJq75fHnt;<{@2FNFTmMo4#&*K%=Q>#d|fDizcbz0pJ%$CK#)f%Wh"
    "e+qRB(d3vYg}$)}2vTxOl^`*598;P7Y0;VBQ4jN8I_WW?Kh2s8j=#3VXUn7~uPyW!o;94kX`p$FW}*M;em@@x`U9?3g;Du;^Q*ud"
    "rTA1EM1DizDWhjI6^SBG&LFwJASX~TZRB&&VVT$ziIF9o#Km&6-}Z5qw!zHgdRb%1Fb#H-"
    "`1X#4GR1gH^9l@bj9sxy4bMOpcX~95hYPO>gP)zQZUpdkw00UKrIv(A90NP&$`PW>0$R@Mr}X60U?2<}?CNu8cHS&p4~Aym>=DR`"
    "9CbUC25C`}wJVmu>k@b{(Da%Nt!UHig%Y*8o|EvLkeqt#(5=zxe!39`OTl(PxnXtN*j^49OfvZR#*RU>U@Azb3>sOiU3UCTb@0Qq"
    "S2`WUU)Yz?T)uLM;`>bPi7DSHfsmBMjLA+89U#q9MOj7aNfA47Y2%nDb|cD+BfH#Dsb0GK3F&z#xJV4<eS8p?AStdUYxtT5;5I;n"
    "CKsK;@<nx8XXQCe0E~ix+E2D&o|yuNBizyUvYgAveLG?Zh)^x~fV4EVmjvc$Gcrn>D;GKqfPU6k1FtQ1lCA>Eo+e)t<D^+!Fa`r%"
    "=Bh_NWnRg}vw)+AX5a4@z?61*Xl2$bt>Eg~=PKiqtfD|5fZAYXtZv%K5?6=$^(O6QGt;fkV{o!8?E`N)G<%SRE6XDP7%Wf0dotTx"
    "7&hwb2%J9x4;wp}g$<~$KZ&GF)PM{qN$2kv(oP~Xy3`6rdnwg%Y(f3%zq)CIP^n{UVgbuNBFexw(eg4g0JVS$XMfPAtR$8wT@)~L"
    "6aj%<C&ECWr@Eh6lAyrn)5M^RVPrD|C4pr(nf6ijfg0nAQ%fjm3GkN$W`U`ksayMb2Gh8oDv=Dtl1|`}Stiv@_V!J#{L1Ja{g4y1"
    ";utZnp`*((0mPvoSKjc*B80iMX3v}x7!zP*SF%JqN0Z4JiL;iS9c^~clV1Sm0RIvDTQRJnAH!&&mfIkNxB{B)D*ss0(Uyo5z|AME"
    "p(SKc@rEx1dfO=v*a=a-"
    "*8M?Q9@@<fi$8PRlN$~mll%ZL&N!O9W2MrVe5`T<0nElR11g@&&QC7qQg~1uI#tM2eQ5(aVgp0_T^$9%;O;VdY%+9W>E15)Jw_>u"
    "OTy^5{7|(!5QhFI5Ya#h3)%_xSY~=9l4gN`hiI}(JdMQ-Pl-BrCG`(zNL7>~LNRJ_8hNz{+^8NN1WEPjMJ8kw9y_S?I_}SL4y^(1"
    "XXsEZ+a%1&u8=zX&0f$v9y2U^!&&DK?>_8;kiZ_w)|z<&7?g00dEIF|w8_lRk=$*I3oLhAS9q9FG${P=>*O8%{j5{Bw1Xt+ULM9c"
    "mY}5vhPomwRSc$-Kv4Xchv#t>K>~u962bN!Rwl57I9$O^w?P`%uwJF?xR&9ennX7`&}`LWt_Z$>=F9=CNTX0-"
    "?D2oJ&;;5}d#g|C#tq|H9DcF&8^tF8JiS6e<1BnXO9d=-cth9-rK}4Ihw}5oL#O;yX}%A-6a#C@;#WLvFkb_GNQmTA!-"
    "DX(?r$m$;hJU>qwvfUk^MLvAjZsJ{;+K;9Q8&;oBJ5iT5&nK;kfh7U+tx>&L9m8Kx&ERo-"
    ")3X2~_^`E!wuuD;$Z0GHBL>SWwlqD!$l5szr-Uys=cq!iJNkUG%Jc-EvaXCgW49zmVk;%C>sy8og*u+{Q2CXqIaKT?}J(6BssfFi"
    "ia!ujjJczSj&kA_bZch<~`m9B+AC;ubPBnu_7xrb^b+hB0`E&9Z|_*Tm6apqwcVR)@57SB86qLQ|XKu+Zvd3}r)VMh>h+!HcCHT3"
    "0MTV@t?4+v8wSl4<i99aT-Tt3y1K$z#iXR<Lw++A<1gpgoLrySLS{U4eoSGO6_`tgO?R4gzJCO_0f#)>tZ+7hM)}hmJcH6}x#698"
    "x-G_-><oCDO^9De#kK#>EfvYFucx>e#9-"
    "(cw>KdNfwITuQpeUuB}`W5kZfMJtWfMv+Z$0c73P{Swrq3Z(MlaCJbi6v;sVMmo$;VS1P^drwI%P$;_lK&%4KBezDi=P?e@Cyg|&"
    "-Q8R}#!FfcL>`AH=_PUuJun^uaO-wS!4#H79n@UQMy3dtrV2kfK&G`A-"
    "3O2si~hz>_J=V%P{VckqfzxI>AQ4UdDa<?Q7UlvjBto;D+d}aK`<gFuW0})zKhM=`J^vLa;sK&vn1~#C_>g=p>ZjsF+pJ`Z$kf)j"
    "Dox&^GYa$J!aTb?j^(AX^b`r;2np}&s$G7?9wJBMP5ejCc;c)NkNw09a~aRSjUujhV3bQp@9nO-"
    "X6Bnrfqy=SdQk^i7Z40twDp5zF3mdAMtuLyzzlrl6zilr2E;Lpw8Fm3v4Al2u09`07p|)15vo6^U1#8{i@QT6-"
    "Ur;5IC8B1F|pY<(;cfhz1Na*n#1w(2&rhgkZM%yfm?EJ2H$mGO!C|E!#X=@D{gyv;xc?Y%x8Z#m-"
    "SP4n;j@OB4n_IS3!x9)PQCSp_i)yE4M2g}J|p2txg>wdE+|s{vx;CantAXQmP>MavQC=hDbEgKfknPQkIpdEm1T5!LY6p58;swi?"
    "R7gZb08FZ`5S43zzCIuL`U!O5V2h7elTHnMk@%^O{2UYd)jV+powZZQE0Dcz6MOs~38EH{P94w-J25(%GfGF-pELIH?|vXMZ4uUT"
    "N2c_3I?@+42)4ZFZ)3IL*6^bkk^7s97yKRM78We7O;)7Uo)Q+4oL#voA-"
    "I5fz58Sc<0+x?V}18HNMgYR}0rm`NM5*<nG^0V28;)fMU38|ORKh2S`rOZKy)@P!kBpv;^T6M9jXK$^-"
    "u9dr;4WUDjnP(APnyZ}+$Y)J#VFW;7KwUCNZ58AWLxy9x@=6FO7i<P}h!_%3g|c_0OJX-zbLj!GrD@y!EFlrqClh(h0=`pgi}(R}"
    "E=6HGOf~=HnE4oHBoEBJktV?lrctk@#4y^Z2`0j$pT1PE(LvB@WSVIr`xcp1?&VoCu7VhivN!XU&;C{a%WUFHx^EM6d#k$V_E-"
    "?k#a>#a14GdD<CQ2V_Z6gf#wV*lSb^+@9)B{_5Gx~lWml%9+a2}?t>pa)9a!<ij+QO!w4$2O^lKN^_yoIp)&njDq9F})iD=POGPu"
    "_yAgd9w9-!Wn)_J<SMq|JN17)*ko+%_(EtQn@>9f-e+)YT}QTrS3`NTMorfGVU4m2eFJ$TyBA{JCYO}mDf$DXn-hla~wq}3WyS_t"
    "AXJmRvb+uEb{$y$h;nu!t`EgO-"
    "C^vvc3f|~Fc6irM4?9#qMf$X`D*6@`yO3=~dOyvm|(5gsKZ29uun=B@w<E+{lkPFv{Y^(}BKebFd8bgeOeU^_o-"
    "2Mk24u>i8V6nc%-i%7o3_>~wF?-n0SF<PXx?F-"
    ";<nYz@lz}#q`W&m%5@$W_?_E&Kz*etA0hI**o$Sf|zL^$a5eX_)L=S8?V21;O752VF_g&g84=LMBQaREzar0hTlpZ-"
    "*LHGiQ${7qZ-"
    ">0+E#idKkk(JSSveBkKd$$m%;fFI#w1xJ9_=zt7OENsfyCQxvS)tWSyTK>w0b`kCvH{i*SPwG1J@2>Og_VGqYz__XT}@LfFvCDBT"
    "8vW`GbvrN3)-"
    "C|mCVDY5#}P~6~70hE>l_*o{@qEwKYOjYQL3>7T{fIh@ky=mBp}{+M2a&8+v?Fcx<%QSmIUhwG|tBS_^a!aG;UPd;^OgRx@ek0hK"
    "+oc<x|AT2mV?6+R`VPEPA#ZR}<LNvq*-"
    "Tn>Py!_ZonI(cMr{zFwx9d>J5frtaL81V{I5(~0%@Vnz`6oBHB`#!jU*Q=2H7}(;6ejPvIZz5n~sG<GhCQ9HV#%kgH(>B4&TpE@j"
    "I9;`|fPyvG*L@|$XDycY4U}(<u}%scpnVvh23^qb0#0+XjecoG`loH?T}~bD)iM2m<-"
    "0}rH!=1TT~>QfY8j9ik9ezUAKP%vi1#MTD4T{~Sx%tpz%YlhqBpPpBENrW@p{U5jzza{H{{^_xtzd5U_k1Fsh9&_s9uawnM)u%a8"
    "-+K@<zM;6SVQV`&vo2_(qK+sA%y-)grrXgH-"
    "@!>ZlHuY^%;Lpb9B*FNz)wC?vElXC{XmDH$IHF{6bHmD;6s8Q4+a@1Nnxky9lN=Kjjp3Ir0*GOiOnP@=b35IxBfQ4zVLf^L($8UP"
    "EtXLLeR<5%P5SsPZM*F8Dtr2Ub_9#lRw>xmVK3<ZqZplnpT$>xojv%kJ;5iZFAPh_A6@c300O-"
    "(}=wr#uimnAS}guF_~a5>qgd;Nsg-ZZZh3lO3cc4SsmHW}og7^M9hW(6@+Yf$Xdj>~mF6!kkrb0}L{TyP+SYw2~frvk337#9HNVM"
    "A_a&%uUkYvwDNRL;QVxeYxUr#r7jU?xb;B|*DV{r{-ClO$QPCONZv`NuKPzQI;NImu060K(p3hebp^vZ|0MihL2z-"
    "AwgfX_!C0lN0x`q`ssI#XVO_vOA}5h_Zw2G&1%+@{UAKpZ4v?j2`YF{v5xHW_FaSkTs8u>K1LtgvdRqlip;(?v0$qj;8EzKX$GzG"
    "8@&)k`RS!#?-"
    "o=)%W>qot+B&a7RE4+TC>;eCDIAj%luw5cd2^D$MZC>kUMuujBhHw{B!IiNA38R0~Kw_N!gNn>pWyCe1KHFp_O!b?a+Z9YM}*tws"
    "o)7lYzzl}bfvD4if9wpO0#?^#Qoxf<zCm^>=9+rPb)H`Zdu;$J!77M@&pJS@MO0(P0qA&%3iB5jQC!Br6JKh}R+6R%A+tro+G?mc"
    "9JLoUN4{`+p8OpBp6nK}BQ##pj9k#uthurZ;LIC8PU&01F$2rX5mC3f(3s0U`c&pyy<Uzy(=;6c{P*l#1{a`A1gj0mp@f?Yv>w52"
    "l2Y9q2wwpma+X9Hp(rpClnYoBd*qNAv`kATrh`NmMuBXAxW*aE2N>LV;<Ok}ZI+MQn#K>8b|vAQ7M=sU$*Z2`D;*`fysU6*3Ler}"
    "imM+M2&XBoAs$tS_5d!F-"
    "4AHliaVflBRs8>|<$N(9?>Um}%%f@fv(|Ict#_@#RZnu9rj?7d>B}DJlYkg{Kc4za*duh;+ZSYnVJ7s&nz=;XlOf5!fZm|MR4j;P"
    "@>-rA-"
    "L}+9$9onX_2fu9>GU%n>=Rgsp*VzY5iQDqK1li%$XjRKZ2>5$@U8ALKF)(0>8+l*d@u`563K)8o6bF}cCYIQ&YRoGu6}`Ugre~OR"
    "AqT`0yOX|UAk=F_;G_!UAc253_x>I6HVvJ5Mzfb_=+Fly1Sxt@MVI2Wt*rd}$Qs_lvkGU#VAaG@{?m4gXuEVTV4QCY{4Tfi6Elz3"
    "yXs6!CM|v7x_ehQM?|N3prZh2UPw+BD}(bM=T^9pP?KQEdxr(m8^$nT&GwNDm8tucRD^Dlwpf__LWu#Y*VF3zwm`hhr0TAU(^+MI"
    "2U^onR2J`qV4;k6V_<jiX4kmn#;B){)(eK_4Km{RZW?v>k@Zs#U0-"
    "WyJTEi&rpY?p^j+BE=9g3XE{S+GP;F}w5#Aly9xeNf!)>}wU6NwFmmLZ?Gu0yrB=0BvbWR%aZXpgAbU%jaKvwjBJ?S|EUy9hRSuE"
    "BC!Kuj5*c%(hw6J9~O*MUv?nc2}y3JvUC;|>A+7gA-_;>w3j`#hmxp&%BR8K?HRva0C?p&=pRu58hD$VS{dD-"
    "uJIK~$cM?eyt8X|$$8QvtrXl*WofD5{cnSVUW!SeejNdcM!uOQ(RnkWs`zr1-~z`XXLb-QXV4ALsyr+a{xRA<jf<?Qn~y&9n1o2<"
    "^MqgU+^xJB<#Gl?azDhf<LQtp3I$?oHOpHQEOgtd0-+Hr8q*uKUm0F|20Q^|SHHwiCoJSPJ-"
    "%Y2P7JB{T@KDhl?HBzOFcBO<Bn7p{rG?96%p~DrCpz8=HH(Ru!YBTF@J4`&)Btq5#!kJT42S?68#IEc=+*)s|3SD1sQam=YnXNng"
    "J^s3XXP73yN4U+JV8Q{ageuA<U|`+yA%0!ZIhX7){w^~5%Y({-cNh6d5#V>+G9b9RkWg^CxKH}M$<MT3F*wX#`5fdje-Nk38N|mj"
    ")Vc(#pNX-Z>X!4vy^N@7rq&}rLper#&A?uxtMGDkrfEOA;`aFIcqzxWp<h#(=%J1F$Y;7mGeV3&M-"
    "$b<Ibm9C$$E>@=Fq0b7lIHyTzTD9NZ=^84F==?pj<y<e>?`*r6ne=TN2g#^iU_@Z5tmmhMKYJX|nyvETcL1_M<NBD?-"
    "rf<6LUh0LUSH2ge!+$rIk~M44Vh36~|zIzjd6tH6ZuO2)dzTx}Jrk=-V*+_g==EA(V-)#RSe{|a^SI9qe``=a-"
    "2VxPy~W2y#Cmjkrtbyva-"
    "R`0~xm7eeT%sb?^{tGhWM`y=BI^>OrCGr`gn|=NuBe?H{5^(MO3sPFeu5snL%=9tTIL+N_=vEVY7Npk-"
    "`wi%>o^y_&Slj+&ABNICMIW9q=771Fsy0tbf+5<LaF-AfBWLY()|1c&6$UtZ4FfdI(1|DBW5dN2?-1Dzj;+V>-"
    "jO^nvi7>_Wp;l{Nw8Kly`-"
    "r4urwuaubp%@#~tM@_3YA&1nX4t_TP8$_BJ|Z_bnCXsIS+O8y4=c_pjs?I5`Ye*wVGZQ@yDCBo+)jh3+zWRBv*BMxbZElK#qmdQd"
    "m+(tqKADs&Dsib#e~XEv0{r&1XTy{X}7GGAnZ5GGFPc&B8I#A2PWv&6>+8WJw%kO0FzCEVa49DJPNc;x@s4E<bi#brHzhYYo~dK+"
    "%ck4-c@>qk}kUVzW1v8L1uL^)>`mL2`Dt>TGEqy3aUZ9~o2g!MY6sx5@--iY;ob%u3%z3;apPbbEmeK-A-"
    "e9_QVOWJRz7xBS+MAu%n`j@w?VBZ++_u5+unhEI5xcR7tB;sXVm`&`tykGhMhP6bccGOV$cQ{8*mID25Mit{+=+93rkgM$l@vsAP"
    "wzMTiPOpn~OJet6CaFhQ|F$q--btQ9OQH=&jUSvFAqdS8G-"
    "^{{D}XMRA6uGp;!#LBuv>g`TZd7sp#j8Bc}?MZn26CTzESHowN)sp(G;J?Hasd{bMinlc>OSoM|LVtmdiaub}Kze!gNX9?Ipqwmg"
    "Vv%3J%X_NvkT|tcsEo)He<SXYz3OmIU88UMLY}s>clTUS&^u`?n8%k-"
    "sMb)MKQuqJh*`Ta4qAEE`dUZk4h1eA<JLPn_Kfj>Eam4F`}b&D2D!;altg#xcLIi^-)%yoq&*4-"
    "A3x`5cj^+wYa<{%R*YLVW6CqM3S<*IbMmc)!ZXzLZ&O1SDW3_2jMDmc*e2nL2_WKT|LVIWbzyuHKKm_HNY#TMZk%&ViZl_@Y#^p}"
    "H8|JFYnFKm8MHB?r_1?NmCRTf5adDPa8qkmO@638+nyz|9jGZEc_L6%gp1%Dr!jHy5=YW<MK3vB-"
    "H4R|ufdM@QP?hsB;Yv5RHdOr#Ac8DFd`fln}NF|Dn$o=YxZ!q8|yKbZV!`QOs7tA(bXxNx)IR(NaeidW(e8140%;LP7A41(SvI4<"
    "ueY*1FP8r)t03B*F$(%E`E25V6=?Qg4+_1kCCT8=vtvH|bBL6SJo+AZ_gE~Mq9@Xo`VOh2dG!&{o4Mx}cv+;}x)>2vvIQ;Pzg1MT"
    "gslP#npKy*;!0dulr{%#5J$sHeI`Xb2J*S5Dp8vC9O(p<K8#Y8nnGOE_rXp%%xv8rV{G(*3HKt`aQp?lqA<?TL$SgG-XCdDi_o1@"
    ">|IHOUVl2=ln2Dg-3K-"
    "fppG?(LxXVy@0l^QUd!^B#`B$N5_OvRJlkY>e}W*_sR(u^<B3vXpQK?E0bNyx)?N?RA$M=&L}k}Mb_s<<VaT_!zypFn1&A#E95wR"
    "wAAl2rNDF?Oz<4<6_4TB4P8AIm}hm<QeOI~Gs>X4=(*hIp{6pxfHSxKF1vxQ}$qQhz`2vMjhKi2xKf(+5d6#ym^f(=(8b2;u_mG@"
    "9%g=pDB{-"
    "!p*HB*59zo@sbGIPZ7J+LdYznU>mY#lm}2ADHlWIN+J(P3CyJ?bP*0ehBD{$7ej5Fu!lUX^4<#@_{d#6MG`&naP!99wUX>JB@=$!"
    "7AX0KdPj1YpC04@9$`_TEN7t77hR8Pc3Qo)l;Nv%eemDvc3btJ_X-ik-"
    "0VVry+b>o2*daFV?_ZTOzKEIA9xS@FpENaK@Vx(6WE{e1yRQkm9C0p4tVkLHtI?T*pwR#p!TCMVdg#MdrOqM(K48RzpN$TKA~hdU"
    "#=Sd<Cqs_5rvO*`Emy14@{G{k0eL$pRHj@CzXvy{!*<=_+6g-P*#$FuXBermA#%%I#;4-"
    "#unzn=3VPY4s?^&H(+;@7hFoXFwt>9?YS1r_|Ygbj-#Htiv$~>FsyhSyOL+cEj8-"
    "AopbZ`QSD)QVtuCzYfKakghyLd7G?PTX|wT(zbj5c%Z-"
    "z@Ii(IZHvs1VU3Knan_sl9i^a}?r8B8jQ6mw%;b8G>6T$hG@=`3%aW8s!Gvg1XIIvdXFI5Z*StGw?MQ_0P2ATwNBNwIA%)L=nn%6"
    "XUoIEnoqTc3*)xF>=h@Eii<Xt5K3M5~K?5__^r|`ER%7Y#PIDKVvG*QanpqLWQViL}D>x%!>Eyq2lCCImNV&F-"
    "j6R@M&UmP=>JvloHZG**TPqG3gHpIyB*JG6tdn#1BoCZ3ed`n%y4XYkX+7?Cn{oV8>o6JS_h7$~{=r#xlyWq%KgZXvQghe$D34l?"
    "TV*eyf1LnOyWZ9%L#G<iZNa$iftGBYxhHbth=+bWCwDUZ=va6i+bOM{!Fw<!c=}~E+t)1p5tP-AJVSw#YXwJRsrp^{Q4u2ha{L|Y"
    "dKH`uYl=@jamhd9VEi<iO9CBSu*9*Z5JSsmGB^X?dWQc`cEUHv{i#=|sS@y!_(+FR^UH)98%G^6W%Mt4;0O}{YlZh#3UN**HcHK~"
    "&ptBwkl1AUxIf4O^Av-urP4KvZlj}ENr0g`v(W550|~?Fo`oE;g@ISpd_%8JIF)U&9_vq<9l)r;Zp3Zf-Mf{Rn4d;sTB=G?IWDW#"
    "i0;8!Y=I_o2aR$5NQOJt!8&`u09uTNO)k%EWTM8c>y?P{-"
    "lh19{1>RK*UupKZb=X0F<3g1eo3?ei$P$?PmlUt$#kXY@FgkPLdWs<-"
    "1)%<!6{jjDI??Xeg}RpSwbD3FeQ7iilCDSG#k@W8)Y;Vy%P#Yuh<C-"
    "nc?99a6ryc7Ax^z&K>~{>m+ScNN&l#a#qY8M<=?*I(BrB=vK~-Ct}L=?YLMCE|-"
    "<>dh1A|Coq0unx#KZ$cNYS9T378nSE`ZmvxQpxH|tUBB|&jJ_1d>hfC%f>l~L@yrDv|5;UE_qwdN~BG~WfoQOt)H=Y9qvTpMNBfj"
    "B)mq^^|nE@cU;$6&y?9(c18wunI=ZCwK(jl3Vqro6>rdnr)-)g@91jh#4MBvo2{xjnVdkuG5Ym1-"
    "k_}7?;pZ>jRorB_dv*u}3H#56R{<d)I48pi26Y~2sJ@*zduRK!$tsl)7vh?(ttYeUr*G2cd8Tt0OZLQw)Ow{qS_D+mn7v<LOd7si"
    "n-"
    "5ZH*IMFR=wbMV6NoZZsSdA;$azZ@@#$z4^k$MCa+K}EYUMS@PAwx55$m$QFZ+KJpXqvP){x@%IPQ84woylMB;!;WsA~gB|JUs2!N"
    "O?j;STbAOTOVQMbtY8CGdRP`HBoUYoTEEHHJbyQokAJgM0w#1L}Jg{{AI?IQ`nv4P@<2#y54zW8sfoDzKCl(3z=p67_N-Uo>8Uk_"
    "k+`?2;;yzB_?e#t{EpOG5P7_C0P5s!E_n<*3EnKxszRB@@~z)6Aq<{*w)WD;Ol|LpiDjcgNzPsPnhx>Q)?}=g-$cUg?-"
    "+K8$|UT)So%j-Jqrf+n>baPp3?*A=S91jQ!4$q`Xl9<P09))srE)qy>wXR+z&)Diuikp0h(S588#5UG-"
    "W!`5SxKZ0A*+?87~Llfx8Y!gY;CF=W6kjHk{$Eb@YFG#sZ~ihdnWmWA{d#zSilN$o3sSI8D5b{`L?pN=s2+-"
    "6Wnt7JQWW`o&1P_k->zcZp9PXO^kvJ>6^lkAJDZ6-"
    "k6nQXXLlaZfVE54ubd<{>RYK3UEil{ueEb_T#JBDbDxJyHEa7gsJLlf>&+9K5{_05xTyep48o5jT1V-"
    "%R&@5;Ypk~Eea9qW^#a(uR1U0`Sb{Y^^wJy^+tJjX=G@NelGI&2gA*CzqiX4A%j-z}Nz1AzA{tT9ugE;-"
    "EQm$%_15rc``xRYAYu`V$W--"
    "(IrDA(jby3#|QKbsjB?}QvD7WOu#P;Ex*7wYgn!5$i%gyHOeJ8>|6vxF=;OV5sBjA+8T_A6U@$#4WX`>V9Ra@go_P2{`{!#Wl);l"
    "ig8jW(|{m^4R6M)pL{jU++3fJizk6y+mb-2vyx400&XG*jn7aBlF`7Fr0GJJ94OlaP9av-"
    "|LKKT5#{@aNkyP}>_8nW`6E7!I&GePK$(xd9i{bD87{*#3*vZH$a`y`tgXk&#@Zh>YNtkj{F&i|kjj2i%h$<>)+VA_0`{_>VgH!g"
    "C!3`FaoMR!#GSf^eSI*}>fGJ{<s?u`H9Mn!06ty6(NPQ{u!r8<a%9R3<Swi;gd-"
    "F`nTq&|qSiFFKlDg{;>*JkV+~!7Vg~fzek?t<0&Mt`jt$C;5$EVw+TW+jkNWW!c`d-"
    "B7)<H9odUTC{jf9qAi9?rC>O@Z&Xw!hZYHsCM0O`hF&z%KKq!W<Go@jo?#X9#AgI!hY>{gpD|mrR=VwKn#T6h<NM#2VXKXz#W7BH"
    "ge$3uFp5zeo?dPil_8+4d8yKgTq7BQw(QyuaK4}K9g3sycp4M5aj+No!RP{*8gO-"
    "*YlS;fUMzM!G8&h!to9#Ph*1s(T+qwrm8B>GC`eEkj$#*x#l$F3I@0LNAulu%E1bN9rS~Bp?Y2JFfz0j_*t^0M95)1!&7y?s!V~{"
    ";bE#*QbnuKW&PNbS-ZIExT;E+Vdj~j+LZb8lD({{mlO!{;CHm|TT)n}8d>hqM;Ac~&S)?7+?segL4mHT8r`os^Hl>ZP47NIJO1T_"
    "?6i2JZHiC!Mw79N{_>t-CUwfKzmR3!)@kW>00ZhtY^qervln431)y<IKI?E_-Dv_EuK>E%*3%g?lP{CCe5xtBGB*P3USq)6Hu)Ku"
    "EJhl?b5a`u<wj`LlVR{4(a$w)E}g0O+kSALYs?uRCOg8;2ReJiQm6t$I-"
    "d?M1>P3Spai;BPzH!WG`!@4>El7_3ETW`9yOHaB}JR^IE2s|hhT2~I@RC3cRrtJ*=K9&26$dC<Ez0Hpu_b&kx;Nz$@gJ#orEjvVv"
    "XvrbfRw<OdEtsUU^)x_P4|T*&3UtHXYX1=>=X*i8-"
    "jh8vAX@5({jq+#unp8rY$eHsaNgT#+h!y4_l9VohvF_=(PbW|H>s1b)tq)s=7-"
    "^`J4PKYf6w847E)bu$HXFwG~&wLfTo%6%lRi#CpadcyAG+T7x7>eozvV{&Vy{oak_{PhV3MYx9S=%+?Qds|h~UZ0<xBlG7sIU&aA"
    "Jsl2}-rn|p-"
    "f?1C8eNG~c_Xgt($o8Ndp7pZke6<V4(C|+FVy@9v4XtRjCFD#!uQj@p2^MWd~WTYMV<xfJsv@yxPQUP(cxJc`_ecF?7jQd@+Xf3x"
    "m>M>h3HVIoFO1we{O7M$4IEIG6%fwHw?Q3P!4)wDQh@L#oiaq8p!VNaR8LKyU*5~#OWycxaM)|s(%ZQ>h{!rh6*_<k3smgPKcNvm"
    "^Tw)rdvMXjude&<B1^L6QdJaz}`G=&JN>A>pTxw^LDvLpOC5d9*PSv0iRE}$K6$^x!`@o0*v9^zVesMV)6DT#kU?yjf2D{yVf~+w"
    "uDS_;apMUo)b508(>Rh*f-"
    "j(FfC7i%i85s#%3P%Ck&g~S={z*i41Vi_4lQf&IAw#ukCMV2A15~Gig!f7`st)^O0{Ig6q2z!HtbS%pqDe3q$b7_e1tRaZ1Vzd#>"
    "q5L~!8lZbxz|M%V7K1G6=1KF7MRpq4jf5j(R0HzB*W0Y~K|lAmQy&jQ|aTA!cIvFNqg3nn-"
    "%PBDw@(tC=j4+WZpVl)us7kBpRTbiHD3`|v=$<ieIpoU0a!#n3G>2diJ6*@f0*IdvB1SBYWr3nZ}SVOosXK@`rv8OP+J2>$3i!5L"
    "d*elj|#+RNOYf4mt{R3dezF;kT!9I_bYV!lBqxWaMw~rQgFy)k@<%@lN-"
    "&{$4aZ}<V$Y{J5?_KtfOKU0pVi9fL_U>)_+Ct=r6^TTv%C9BtzANA~S6Z1s@=xGv*Ct7GYZ~cDeY4}M&EktoV+a#bRBcj#Y3{(;&"
    "!_w-ay<zjDGI(C%j`uB9_aqWYuxdZLb~Sh{;m#f6-"
    "m&#_LNAsyn01_b7LOyd7W>?;8j>pq*a(x(g|G1du31jj!gF>0Q}MOdpRAbB_~H}Z;BF%XUGHcxy_@R<t&lM9ff1pzxGiPU_ULy-A"
    "j)sse!GwZA(se7yw`PM8H~a3E*=KUDb{z?l{OeGU)9dM48Lc$4$Lm0yWvKiZ^VQ?VKQP>t01Dp~|p%V!lqQ2x(`Zv(eT7Vad^H*6"
    "NKT$i1mp*+w;sEAd7`CFZxy03`mdj^)EM+gPeoQ~pdVkPFd%2vx(fplTNe2n>=vsbG@}%PB;H4PDu)zAH%@WYJG^3Di<+2}4vnHK"
    "w32mP<@Aw2y4^q3U6USi%>rzuKt(O~eo}UF)_lm8FT$@qFY*$SDdG8#@Ez4tun~)qD~c_xsJ0VeIYByQxE~=-"
    "SM^MIN$tKOaWwu<5!YO{#&0HGRcH2n(3LoNC+F1gB}<cls?cSGO?6*7adxB&_3|4DisNakW^fmuGz8YqD8Uci7<jKWX0YQ82ct)x"
    ">H^w<luqjy$#BFtd7xzAyPRNMGETTt9z#iP?HS^5x+<#(>UYwMToszaK{PDDUy9ZrY2Eq0!l*vLi2Bx{Ehx2SjAS=pa4E^A?3R3|"
    "7lLQYS63b4G4b(;e*oQy;F6YKi3KvqhRl;p$?|Sxf<EMSR>Xo=q!a0c#J@TTC7YzDV;7qnpu!Z@GImPVQwc1Ow#@DS_;v>tym=&<"
    "ABAS?DudEF(c)Y+ABO9SM#`V6tgczf^lHc**!m$U<*=wt)1V{{(G)9z13Z;E%>@b*?H#UaPt2W9#$$9Ywa6flezdyZa30{tUoznK"
    "&zM2FeF#+}qU)6=<v=5JFdB4CqyX@Pl*j4^BuAU4QTzUCq0jt%P_t>PSw`msJO_`|qE%w_Do%#SjXp-ktp&ONgPKX1{fDiI~nw5L"
    "ZkU6(06oZW^Io$qU2kNmdIQ!TN#G4Vvs|DY&-"
    "Y*XdoC<VwBo<mM1yKCx3vy!sm5NKKqv_Y9Y8n6+?LI!{56qP<)F%jUb&nRp`ZDEt6XukLo<80EB(jULgrTJ64RpKG5)aR4+(f;cP"
    "6H&C^3L4r}}?G}j85zFB-8jn5Mz_zx_^9Ow-"
    "C|{u|)$Fb0O*?dFL~?f7L`A53v++;BQt#RcgJ|v(L)cll9MB*q?zmn=i8GB`)nC}|o(zPIZ$p*u;S6b^4RC@is!#D2CcS`~b#F$^"
    "b`Y#{9WJAcUvfw-=}WM?nzT`mS>jiQ)VU<1q?ibLxANMapeE_{X-"
    "%5l{8&GPzI0W&ziwGTyh*fF{8p!Oau>@Lk!oK8oRDz@&RS!>BM{vZ58&&5nr)h%MMGVT%rO;Nh$>{L3EuAB1ywFt4%tq8qc8o>o="
    "tZG>}tsHdx+UVC9YQ)MBcpy3_&K>ZXF5o#@{k=H5XkYAG=g$r2|=68=6ekJ(ThU=h)U9JC_iZCYl&hQJ~dTbDXn{85x=c<MX^-"
    "c2fTfaZpZtj2imY9SDF7q}PdSOp2A!zI^qBrs9s|khSb7Pv@%^r2svnaIFGxy%XNLhveUXkc=BXZwX+`V650eAyYe{uV=P;P|qg4"
    "#dN4QDh>=2`LJtR44X?`-4)N++6|fABMH^s5~LcqW5-=ue~#{Ywi1VJjt?W!*##Qv-"
    "C?x`V?Q^E((w>O>VMI<)*eoM=At$%6RpfgV{oII+(#)(dZKdC44n<S(~YIjpibTAtx5{76{`Z|OG(@E7_QNc&(kST<;)}o{Je3QU"
    "SMv>96vJ-ca7VD?<vifWB-nXfF}5LW;}y^l>PE@3{_Twc((qiY5lR=*j{GbdvZf^(k8PqE1Ii;gK>4FTcDBc+aQr+TLL4bm=$n~u"
    "Iyj`h2Di?Rq$`Syt=@*Sr5vGp7!3`1<b}xVfw4l7$^FA%jLw!LVfynToeVP_SCdNBH2fMw?K@Sc(5zU(H4kgq_%(QTNf$Lo-"
    "E(}YhCk4*do+r^Lz4IGl!clO+u%){>}T#+S%2vP4gA6JFnRIc8@$DII>6Dt@@9?mJ^&LzJf`4U~r&6M9u6mp&%9oL_%lp21*KbQj"
    "we~lkx(R;aOpsB@J~&?8KG~QPu3RbOr~=E~5gb;MT7GHQmFJuPqG7x{2k6KA$vs_B2R@2~w<BRF_F`X7~ZuNX-os&t#EN^><nk^+"
    "bFyN$Oa8Y|n&*uCZ_LCqX80WPjY#1I}U%3Udvj>T0#eV?X!%pH^%~>(+i3F;A&z>)*rOrkaF3(F3@1x;68k%le;4U_C1pE&}y80k"
    "OSl&YpQq>W$cxb0VG;EsAgA#0rbB?-WJG>iXqOhRmSNPn62jRNBIdQ-"
    "Y71#fkB*OO+ef(|DmHHjIjr9(Mz5!9^e9hpI#HNemXmtm#0hUh&_r<b{*`^%y=kfg=6jNNZb`G+_qeqiQDb1b7GYjk~ju<Yq-GM3"
    "J#K+s_b`07e;p9*vzPTVpv$M*2vrRI(Ysq_HlU=o(=q#*|I=KfOY{^TOhg3=xirfMq|Tpve<jlVr7KDAkWrhHc_Z?qrMK9pU8Uh^"
    "y~Uo*{QA;Nrv5y~;#6#qsLailV;xGa{6o?g1^ZC0)>~zLUhHx9}i~>oRyhz5Lu9BJ?%VDVBzt*kA>S@R|^HZROXB)?U)u+i3_JVL"
    "h>*cCmHCgg@9&YfgxsgQ<Z}2}qneUo2x-e|2oU!-%67rsP2-Xj-"
    "!or5)s&P^E5_E!MIxcEo&<YcExO!a69(&Su0MYODK>Q)e9s;gd<OP92?j?GPE0ES`kK615yQY%>`?bc8KId0pmCircRL?48tQGF1"
    "sc6dpq@yP-+yx&AU5jwlKa8dV?2IL_VR@ogsVZ7}FZB|NwLzknN8A-"
    "Rqxc<0RFcK#gXqV4$x3Xbl(bPO$Yypz_jCBNioDDc7D|5`iqWbZYP^p?CyF%#?F3PRimx<@?X>qD+A{@I=2>nqghXF>uo@-"
    "4WXSMBMe^lV!gi*HG}1FnyH`WY)g5()eOzW#n-"
    "b1kYFU((g3m?Rv7s(ioHF=Ku_+a(IE<cG{|<%!1r%ti9_AEn>Db0ZSmca^{CZ4X^4h(L!|`kD3&cnF%dIqcr_YRS%OEr$0jR(@l|"
    "=h*uaBII=ZMr%`Bm~e@ssQA4%je9$vY{G5|tFBSRw?-"
    "z$a$xY(jh^JfGZ)?Z4h5a<QNX8rv3=V%8xfkpXWb1u+H#mI{{zeEZjq7`@$~NxULB1gXC|S0PvZu+q7|;$ppNqU^<VQRrMxvd)O="
    "PzPawJ5pi}O^RgqU@N;kH}*nyxeX5gufObZpc@|<gRutz9N%}nL(N)*fL6s+v2!bmwa>^`%#XeKhv4kx4}MXmqVgJ5<O4qxK*4kK"
    "J6W{U#0hQp@9t}61?!`okLd5?}i5tu{Q>*7g*fQ2LJAx0%^g>5%8+bu)1-"
    "Bu>dotb?e#5UUPgV6EgUklJnVKZzXh^vTKf_goNbsfG$3UKGeg!+b_YTz6qK~MJ}T{Jb~FlFrY+2jb#qY|NWYjm=<VCg)AZ|T0CS"
    "7IsGa6kG$vl;Ne$weN(B)R&p{kmHo-oGmO<N_RDi7nYh$gN3SwiaSAjrt$#%`4Q(-cV>rblwQ3t-V3b4<K4YIvj=Q3Ve@ho+{-"
    "vS5M-I_3|A_XbGuI1KC)YzY=T(oi}rQXJ@QWQ+p(AZy&3+lG0v<gHg{^3jj++Ip@T-"
    "6C07iLULylyjYtHJ>;+C3E;ScO%tjuiCYE&QFXUjL{`wgdxot=-"
    "GF5<4|<|cncN=Bf+bq;rFXLSz}i$Yk&DfzTmm}fhv;xe&hFq)aEtiQ-^mUMc!eU7-cR<-#Ozq~3h=G$yzZpIlP}i0bM8YNRah#t-"
    "D_er*5pgw)*}8FGsdUA0MN<C{cIXrv@<Bt#&li420N5=aXYLkOs!b;z47LC_K2H<Ya`=H1wq|(I53{9J+g#`;xjlIc=JYG*HAZCw"
    "0mCCml$`_wSl}6f73j-(m&{b7$l=tZr;@IdtCE7tlj+?z-"
    "A)!``SYOZodk8+b$_eMNnEk>_JC}<FLcqb`yKUk~*&@>U)oqfqw5ZI9iqRSRIH{*Mwk`AH_dQg!Bv%1jvhe$g=SH9;Ivn&LpEDE4"
    "2>}=>PV4+J4nrl*A9(quAk(DICOlZImk4(By4j7wDPg=y|L8pa2HV{<C~>O&$#{SakB45=jrZ2wp;noru(&1i7|<e}4XxPjIFLRY"
    "yDhEoogy8Kl+^on37V@R(zROYUfM7O8dl%)_P^S>i#T^sF~g0x?}SpZFAyvei~_8<_B!B!^e)cx@dNa=ghlGM2w4)Fx?JEanfxXD"
    "b@4!$ryxRq`(kwAiWUX-VeaOk#dtv(+k`V3gXm+Hy1jN@muV&+JY>mi1jIbv2MC7mJ~Lw9JMuOH>Ljf(~fQfyyP5<cDX1ElmRk)7"
    "aWwOV(Ou7{#U?1h2!UxwCfZ0+>hm*CV+8Kw)Z9Oslg-H54gQi@I!ZR3XWBY9J@K?wJ6+Whya~l{F@HoA#3M@kddf<X^a-"
    "xyu|^Z!GU`1q=o<RuDwM@IrT5_Py8lt&Z{y5Wz)rB=NgU+Hu5$ABpPH^q!kKm)t}M<<>i5LuUWYM9_{y+r5snDRaCA<fQ!8A>=g>"
    "`OPa%zTF;2%h?lmqfbU=6lO~Ikw7DNZfW=K@zHC_#EFsszsF34-"
    "ijdF5DIv@t3Y#;<i@fHe}_KBYG)DP5$W)4az)$_kMuSkLJw|2&hYYQuJa}sA+MhCXYbiK&~ih=$LxbWG-"
    "^HmM}AAz$2@3Mt8d4y>8SmDR1>9`ICGR!mBXmD9GFlel-bC?&S{;K$wY>7EUmcn_yJ`6e#5I{CM$E6Sypt@Hd6|(HhVhe>Kvw&vO"
    "8iDT3&0a^wzL3<eZ~U-eLnV&2Lv-h68&%<Wx)n2GyXWa+f)*T@FVhd@}G!NBArBOFKybAKjvhZ$G~sL=qGnzrUh-Ty<-"
    "T)iqqLfS%_2p_f^ba<F$0wkDOV4%<K}&|N-"
    "uN=TAPj_GPRI|4Qi>yfkPY!6BdsS#lAc7K(xZPc4t`FVf)No#MT0z;}!C9C!1BTjX>KTd%txg|NM2FqRhdD7PuE%DR#{*Up-"
    "442Hl)6312IH)L&i{MmtjTnm7fbQ0P2#+=sTOOdQz7ST+u0UYje~YOm<A(u71Hok5rJTS~Ygs#my4P)qL2)es#S!)Vn^aku-"
    "L*#`hzP(<Q{CI2&W46HUOIn=06GA{2;FM_idntzWE9Z0mg&r4cX;Q8+)D?ad)iyy3j_x~W$hacmIAvi#UbDwSI`s%WLk4E{{Bg3j"
    "(q_?KJ7aJK-+wU7aiFhl_%`M)%)puQSD@}Rh{?YN-8I~R9%Dn9-e@pWv-"
    "@1i;_=kA0?U%7b{xyGtgKXk#;NC%fsv>)!`leZT9z>X+8Ho6_?F3*$8RF-T_KplGTuTnO2(T<lu8t(mOwfOSozX&M;H-lw-"
    "+aROP@o5a?KLiCan`Zpts3hNSRGhZCYo?Jk$N=r?WkIw2${3-cD3wWHyf_3R-VYz@G%%=O?t^gyYK-!IM3!Dt^NX7G#mNLY-"
    "~tn%V&X;vaO*Z|d8V|h#bkIZ@_<Ze8u(nCbQT2Cp3yu|QJum^Sd*m@Wu<Ov>`hP#x}CReJ(cu(@2us-"
    "77=R4`PYmNf08=J9XD2rRhq@Mn545P_!p%6WS5q@iffVL(`QM)<ffTWM-IC{^4sV%h%qLH?q?g9}?m}33Re4<$eCLV+!cIA^c*(="
    "_VR9OsLP4WguO6L*@Wvj6|&Moep{-"
    "bv}31mbYt{nZr3tkJ%Z<)N_GZ=6{O2hTN1qNz2MN{Yy%>_*$r&{?MR!+ncKuMus3~@rkxXZEF!tXe?y$R+9@X4>gW~w6$!5DR&Lu"
    "NKN`A~LKv!U7|29$>5&<l^fSsCos+wnnWOUs^b`V~aqtUvEyi6OmMLwXY@FHV$}W9fGiY@J4WE=jbM|JV;h0Dp{IXGLB$oZp_FJB"
    "l!VI(?o<KMl$0-M9aJ7ZHW>_O4yF87UGDCVYC?NdVg=0iFm==Y9<#iAI8mgcr9zI{^h$$CSz4m7^eXW-lRj)<#JZ!V+G#v1mV?Ce"
    "v5ATAIe?>iCcA$Ryz;pBxX=k-"
    "@4JaW*V7x0B!CKWTn60v*ozBPyzrUAS2GfJd0+o@eL;SBzGx)TR+vQujv)xpcr?yA<x2`;0k`ZVVM6>G?<v-*o_2u&q7{0?B=-"
    "Jr9RlZl2*Q25c5}EKC30@&hWx*lUxGUeobAnT2-@?2{2{tSOH-"
    "39!#TP4O0et*6#mP<pZxk={IuzJ2YBv7$)a8OWv$2PHB(ysNtMwR93nhur#MA=guoj7ypT)P;M};A44vB%`14?)6$k1M74@-"
    "EK2&y3XVU#<$S~6z~Y{1c^7DzYLrZjp9apvrfdJ6!#~X*LLB>V4h3xUY_3g8sA%W$U?CLsmZT92Js}=v{pD+sj&{Z*=pWxoPQ?{&"
    "xuyMA$A8tSt7r;X&*Xy?$LCe`>f>QEPqdxz$$wC{B8!rAm23gCF^)NvvGKmNi!I$-"
    ";)@ar!e$x#zgD=9_IpY7G;uQP*Z76p(m>s?lCHjD{dRI6se9DE~(KDgpTPl8##h=qGkK14dn(Ei>BpT2V8yx1N*mus?{RpbAXBO@"
    "!1m-"
    "b9AQw{oq?}5sGd|@*a`8aRLZN3ZF?I6Q5$yfm~D}I7i8XXj^ka+g0L}`WHvxaRy{bJQdIS+Dp=SjXgz5>Uz}~jj5^M;Ql^j8(@+h"
    "e7rJ!%$O`Li6*7;>3_Xs7DxE5jgmH!3g09MgSU?vgi$~dN{70`l57<yEKzO;1t1RuKF4lQy$RXxQKbmQYRA>x$+bHe{O+-"
    "bY{{j|@3XxP9J><jwMQ_!?#<>k^t8$Lnq4!jy%AzIjJs!I=<g`>6JoZnjRvw<GG>dEeX<^m@EsWBc<mTbB(L#&XRNsPg=TtqV~5|"
    "#IqmXBvBOxa=v?bFg98nzF*=lKt~Z0mQ2bK?e#Q;*7i7$%$z)4j226Mfo$L4~7$44LbY*fd6jz^)_?|WJ#F+q_swOV|NalH=d>vf"
    "WT`$A=-Z>}vlYWTb%XsxaBMnDAgGp|0EP*H-vY}S(%Ik2V7-CCU2AxaQ_3nkQZ=r1He8Gf=U|NdcB@7(^-"
    "R{PKl#>Tz3fWXx2+%B6^#^0Y2C2g&RH=wvYxDS(@B%}Fq_Fm?VoA6gTe8I7_W1%mI+UFn9~?nZ=)dg-"
    "*uFI0jm&}0<=)4#w`6{W{o=5!2bCaJQa$TW`-w>a1yx7QH^GyJq+5ZTS&%!9+9$1`dvn=#-"
    "AnndXHxZds}k8bH2fhvs$N;x9oxaf$Gf#=5TusvOfR^Ncry_P38!9DP__~kTg{M;0OK5f#3Qcors)y&GIr!0WmaFZU6UEIN5XaG6"
    "j>%5nXTU=ZA|yp8BKbdBt=V%ZFqUGvlvZ^+^DVxtdoVe)fUa`WRRXks;8UlU$leZEs)kAUnd3tV^*>FL?~2Bik*AOn?Wl**LHMBU"
    "W08Yp9PdivM$Z{+aV)V-tZaW;%<G=Yncr#JhMIHv<2qcGWoebF~O2R53ak%o^-(d9-"
    "JiJBpYscz7Ba>cb1Q_MP{es^Uy+1{uMFO75!IVGB4?-"
    "ol?eD9rg2=0i^#p70~TQTC)3FlQVqGCM|gmM0Q8w$2+Y<PjLarF__<D+=kP0w7(gj^>HFJ1nc353aL$X%5di;ZCY1*W<oS|_8f27"
    "S&cU%i<b>^4H`bB-{zxo6o2%K<QdtW?ZvdKE+#~Ea~3aGj}S9kNX)?Uc5Ah>_a{GSOEyU-!io}b@LMBKKmnf!%b{~-"
    "ic0y>Y^GO=h@r0~oH9gS4dXV?w}3{NJ%+KCrtwO=<nH8YZN!i)@McQ;9+OY@SkIYX;XhqhN77LiU4cN7hcvhSSGM@2WVZGCKB1V}"
    "0mGE1@jO1)crXd(>yYBHj-mL;!?Dd04zqxda<u~djG1%-"
    "CP2ih&BV=Y;P)@Skq|teiY<$R_&5s)i*;^856GLSpc+lDVbS@R2)nNid)t$Gz>4q}gdmAB)(Uq-"
    "i}&v6H6`WY@>cjxuEF&!5#@3=+$#APQ9x4})<cxR09Lt7f{rG$X9rf{-"
    "5WxxXU|CG<o|wJw!X(*A7J@@wbIhI+jK#6#Mx7omw5Z}va*!FU+5`?CA(B`#_i4AY3A&I9%HY`dfU0y;zz&srF&Mck$+J(s%2oL2"
    "o%M&)vBMM;PJk5D7Kg<Zar&cdZ}*0x8(N)6AJLN4{iX3z>#m!+U=A3wW2e&N}0%DD8&5NWz|DT3v~k)dLb#XU+9Z{Sv77;Ll)#l>"
    "={n`lv}&t8cBdL(8_E$s36?FQ3ve9+XCl#;-"
    "0pF_%ePELbXjCMqSf89m4NX8;nVIK+;`|`kTg3ox}O$99fc~*q4$7Nf{#9vUa4(nU(Ny^mgN;4>4c{QPT~In)-"
    "D3)5Y{&2aNZ~EcSI2clW=h!XQ#q4~;s{id=aD+&M|zwoylnQrlJ}d;@4l)sg7avoG$%s9HRQs?;Fh{me%o-"
    "SgPS${KXg8MnJ)wS;)v6QdT=G^KMF(z10!26}YP^(BU>Bkj;zUAsX<b|D5!e*Vw)y`_#enj%*|JJ;`2t12}WSfh~LE25nIblzn|h"
    ";p5?7|$wijfFlbVsD4|U14f6#X7dCeIU^Rp;`hR`9^=YN;A=E`!}2M8X_Jdf7I7#@?~wMSYIX!)tueNkVuTqX(BspF*V7?3^lS*("
    "@7$V6lj*!COX<?SqLrCCs@(XGkM&NnkS{Npl_x1L=sv)t6beWu&*UL6FB4He=STtx1HV)_mwBNmz`u(uihc9w@#If9m?mhTz!lv3"
    "8bxg)TL!k&;|5RM$j~CV-"
    "d2FG+;af813elUP+0Y@T$87$Lh|)Lmls!m1P9{3g$PcuBJSmFaWswdjyCr!gBrY$rkpk_H&go^wIX*oiy*|2s84G@Aht)Nm9Y~H%"
    "Vv1xgZ>5Jwk#2W0+hhLN-"
    ")<|I1dZ$I7)FA)`MNFdbQGD#Ul2^+x<ug_v)D(y#p9nIr>2Mu%&{uj@P#hwIn|$Qx{iCvh7A#gce(RMSA9`{tmbOI^V%at>VNW&7"
    "E}J?aQ<N#7I&ptZ?;>naBj%ai^D&3rU~t}tN79|Lp^nNZOZu+2En(DiAutuel+dGk<RV0gXVYG=<Ok!rUhZI_E9@4lj#>bcuW%y8"
    "o4S}l*PDJdQ}n08?uKf9OLJQq2r`mi_z8~FJttotps&w{MBA6TEUQc6LlOVn`_H?{_qmpB`I1k4F9kCVXpsxRy^CqaRf4!^UgPxW"
    "M9@zGyV#Ecr6HPidJ?td{IXR7bminp%+fvefU4InH$w+<eV^WxfcDb$2J%wx-"
    "*6DY3{ISSz531KDv)uZ_xL%@5MfsqMFGNyjRAM2n=4uvqa_Ivp-B+4~Dg>&mOi!1xVK6W-"
    "g{Sex<dY3_K_`15#VF%k>2;{o|^oK^jV+yMyI%o3is-YmazG_h+_yw)(W3u1reG;^ch4}z_HCWzSd(Egc2I+fdn(Mee`<uNoWX_!"
    "5n62M%NJ0y%Gs*GoEI9YTm#}5OfJ;(+ND3OorQv*!$P&UY2#7uz4fe3>Z@BThkz6Nvx|FE|46ZX2f<1Z7+<?FOS9DZ(1R$*&*=p2"
    "6*iYbob&4k;D}8xlZAyOao{QP)%(Hn+N9!8B1E{+)ac%uI1qtJ)@W1^FNm0ogdt4J(9E465F)Vfy&vQh0PbulT?~aL0+s~5+_3o#"
    "C3$ZSN$8S6Z{k!kk!u%eOSP4t#2yZ1R*&4E~pIZt>i3Q}bH3yVqt{#6KMRx}&J}Kjth8kWU+cVsbSPR|>#4MKb&hRv&;T;&j+2eS"
    "`e(#@sPu!ZJwhpvG|M))=I<JhVn**?9HLL<$+t0IR6?G{~goQOe3{J*}N(k|Eq<JU+hI;_@)x;4NdM;o^^TCfZdZ{!pFsG5tlXS7"
    "#;-LNBn^K&_$48byrt-8h1g=R#V>!OG+IyjxLpTxveK%i%h-"
    "FDZZpZW``dU@%YS$1#JKix48A3iz^F}J{>HMAk*BJTad{9DbxlAJl=p7B6RmaxQ*;K#!L@e{~n^YL@sVW2|wW9q_65Wp1*#raN?I"
    "%G_-lf?YvmN;`2+rCIlkw*pR`Zb@)moP=0Y`*r^!C^posF@8zVXW`cP+-"
    "@mKfU^30d|aaPa_cM|YLUFLZ)B;=U+RQYS&<!ccp`OFa@Vo8Xd9Ku9-~cqq!^mJY!=M3PEh_I7-"
    "X8RVsaw)d(yS&jzf^OZECbNL21o9^2+a0JEJS|WQ?_bXIn)hmH5<9f;w?%Z~W_%$m?s*_&2W}!$rnbktW;cOz0IP2(0YT<19NzmZ"
    "189y6aAM&t(gmwLrNRc^>kc^wg<gsE0ye2i?gZpDG!Z8qN*=Nsma4D`k1=`j$%D@Jq19}GxW1;>sr%Iyyb}8oH?2)u#p9<BABz?y"
    "L@jLs6U=huyuC5I-CE>Cl&e*nTBBS@F%R%pZ`*Ec$)r;q%n#fZ;Ct0McV__NJ<()_(Cga+jR2G}ZIu0JzfvyhtLyhz9kmu@t-"
    "_PI#tlp2ouvvx^#O2vc+HkGE5`9~It-U&flJRY^tXjS|p(-"
    "XM_n7R7k@{Hg=;%`F)uJo_wI^U{KXX^yuB$DRp)oBagEi=;EuP`}G|9=#t^XzF`_(Fkb(sCvGFJ^@a$*p+#I4r6O;~Y!!M9I&>;d"
    "bVX%EuPay5CFBT$#%wRl@}#>~$OKzYzq$R(l|BgDA3CQnpvx8?ayY*7E@M`pw_8D_4y=Gbz@8+0{B#-"
    "v@^V6Y|=sx42;4BuLB`&!H2qnG?1>ZkS|@<Ut6CzDhS*n+*s;gd+fw;6KAgoGW2=6blkP4N!+nQ28n;ZErs>F_3HVvlZ6fHEJ_6x"
    "ZqFwMf)1+dg`tD@0Alv7C!`LsNZTp_$!ylvD680oUWWH*iUyJUaLteaJGQ%4a&RJua8$XlbQ>%1r^CHLm(w4RS`f`gZ}jWetUlx_"
    "^aC2Q~bTUs6+tTALCH`OY--I^(7U)PFKjVF{KR_n%cQaoROXDxFNcU-qANUR{xTneWV(FaF}HZP?H)RW7^XrB4K_9wcZinAQh|@T"
    "*p;NWGjz!l;ZEiG^n)=z}<uhJg7vDmMb$D{^-"
    "}sh%~dWU<>2bH<3aCb{?u&Wgo;g1fdX|BdkLh<SpGwP>)!yT>%CR&gB?Dcv1aaAZ}Wy|~^Ht8)?ARtQ4gCRCol;CJc!fv!ee?e>9"
    "MgX6=UVl^?b1w=Cv?M%&uy^1BD30oHNwm58ITKcl_EQy`NRytkO-"
    "$`^+{*k~<CN36>1b25@hgWhvWpb2hOi%RveY|y+q`(+G56~}tsMWTJSbH4tQ=Of~n(mh=>&--L&Zjd7cPqDK>>ZPmnYiM+g#Ue~u"
    "<XXZdFC9iXBvH)nl15Dlg!Ld>w9DW+L{o;4w_(bxTd44-"
    "ZgQSq~SswIzc3{y*D&!VqKdrVtBiSq{)LZxdt~z_L70P6v76m@$qA7J4*+`5-"
    "X$u3A^IfNxf{tma`W|K#TEGtCL>7m?n46EOA3J3SOh%o-"
    "cl1BNf<FEtgF_b?fh=tQ6EqV94<v8_e&t85ASkn2p^I3s8pSHTbohM8sO1K&DZ%C*b*+JrvDfpvy6_c38Aw5ulj4z&aljGmV1hiG"
    "X1nwauk=CevFcRT|fqr@;G)cUuENfnD@+>l=wmjnCo^m-M|?IK376ZB59ouXY&;<D(1M)1A-Sm)uSdv~%-"
    "a9o?>S%%tKt{?7Zn=X;*xFYE_=gmT*VbCnW%+Y_=bi~N~j^k^(Epkom$6Dsnn@(A7`;Fe)wSYo<T{z!;{bn8Lf=2x#>^bZRt+m<t"
    "ZrV)vHy%P>{?(+mL;Ub#$m;}u_jysryK47dCJgiD#RDs)~r0QrNDjoBa#Q7luNm-?6G`>3bx!*&QI+A>OH9m8%QAD-"
    "?{l!v@>0SKQdt-nc>546%nA56va`x7VbG7(fo(>B$s?>LRUGDQcG;a%JHQA&l!JtdnadfzNJ7-"
    "AQ=Kz<U)}m$)8r8`wOnyp^Yl&CN0O<^cwY%ulPM~Eji2jWv#%KC+9h1dNZ=gqtet6thQ4cBwR(i8*G7<Ka7pE7prtUkjyZw0dk87"
    ")T+@nZKs@Ls<;j7<-"
    "{VLdcFMBEDz2_tbT_wJh#tIn%_dkqCrc3^pF+}$``wdt0S(2@|{dx72?+{Kwn$|S(^`!gDV@GWR40MeHu@2crJR8zfD7Rqexnosn"
    "aPCXuWX3w540veK-|aa!;2yO@LMD5&mZ7ZqVmt}!q^d-@x$RXieo0?@qz5c~W?JX0Z!%-?`z#-"
    "$Pjkzqskq!G>^qySY+m6d+WF=!xZ#<N;)*hw7-2*4WO(N-"
    "IAh$FA07`#Z<vyneMZ@Gbts(6LYu`E(KKbGURL1*g$~W4eDDqx?@JMy^}l9%s={6z8n7V0SnnE|b|jEf5P+8Tvb3#isdrl9^y<@j"
    "9axOICz$uGB_%dip3aKeZMId(4HGJS?$`II-*NT1>jw76us~|I>n-"
    "}_gWj|?*W;~YU2q_rNfmcG&LtJ3R(ZDhzJ^S!4zZ%?c!puT)1n`FiOefnDEcNf@+cA4*%M?sSA%;itzK6LzjVbN2<>@{YMVew=KJ"
    "qvTc>LAX7VnD_Ix$ABOuGPiTmuuX}4rAAF+acL<=!EE{r>X{E|0ZK}LCK&>8<}`0X7bwedvNg)1zl<E%A3<Hm?X<S}8(fT87V-"
    "jeX`L>hFf1v#}PMRlYvzXt_3;P<<va_@|<yYf{&zuNTXTU27qQR@J}c@e<OZY$+uBrV|2fP?a+Ysp2wNNEzuloI%#mJTe@8<U*B?"
    "c7+?7LlPEWYu>0Nw81YX?~Eg?SZTj_3op~;Qi6<pZ{ii5N^v#AfD9`4DMUjF$JRXw>=4K!j}@7njFk(fqLhwQaxq|6MzrF3EfYdx"
    "H&}6u$%M$Ujk+|n>*N$>GviMep&f)^L52eTXV)O;hi~3T%lcxMBm3llf72QcB~ND<~CMp-"
    "FcGYdfvD&mu4;+VMRWsVd2$l?^DwJ+!?<+tJ*I(LAp_Dt9$KD9%r|`1Lsz=CEhg-k2hfax#WMJ3%~fCIJBv-uNX$SW$2zent4rf5"
    "Iyqy)xFw2qL4kcb~`y*^%BtF3^i8M5rb~&5byp8N6(1g&9&h$x|5ey`2=Ccf7~BjV%a^wev9_dBwQ!yd_tXiaY3a;LccBDNz_D!f"
    "BlJlePEo}sxFT!u?S05?OVkeZ7CuNdcUet8K;?QP(Kf>_0{h{w2rTJK$a`AN&u+a<eH4U-$UdB+DgfaInNc%Ft-InSTa~aS67}Zx"
    "#siR1Q`yq@H=Lomb2UP5IJZ;YLxmb@Qf5M%D<zkO43#u3)2!{>wFY@MIgtTdY+^Dxvp60Gd!F)UK2x*22kL08PqR1;;XQuQ@-"
    "{F_=gL0g*mg%s7?1pi}pm0F@e^5=H0(HH6!DGIB*EJAW_dxubcSKI9e^4?}4)0*n{QTcyz_JC&P-"
    "rur8J!MLFCuTTX;aYgZ#)+!HgoG^6mW<N+!>?^eqp*2n<%EVnuzhdY+N^Vvd!n_m;_{4#Yj+<zyqmY6?+PFTNQ9N^0ANlCcJ=*_r"
    "thWChK(V33BO@Oz@p6Q*SXetLH*C#Z!7K#br3SY%>wSAr`Sy?b6QM7jbYNZu43Rd()awu_K?Zz<u?;D`;@S9ZH<xki05?-"
    "%Ku$W}j#H8V5bMZfe+y!V}vJE_@OVn;r!ILN+9qj(77<gCpP}AHXt6swF@0qr_E+ik3sM@$p-"
    "R@M?c}GoCZ8YOFkE&(QI@o1%Le>Ln_+uh_Y>&usL_T{#KTyr)wN!ZgYiGYk#3(@Xsi2LNqmK8)JM8nmBqqRu+ZBF(Cr32cT+B)2o"
    "=8D7Qmb@&sCI^hU8|ThChE!FPNeR|j2C+x1i5L=F5~>_olhVDR<QD2TlQ*Z>jr#k_urK_6PZ){$~D#o$?Wt50{<G2duwAs@-"
    "66SEUdQgZ6`L`lM}l&=qaHFYrOr>`6+qT#0k*emYxP@Jpi=;0n<+1C6SUM8dV0rO7s|b{+j0?ir}ROknEyVP64Sr-Fj1mpXOWKUH"
    "xYH9FHf|(XDO5XRreL;>>=ZDN#%(e&r#v(K|5IHOc@dSjR+Z0mvjw%g0!z-Z!_+nC%>AsKv7vNEG7Ji#FSAVX(ei;-"
    "*59sfwsn%qq`*(|E)*t%4(#<8Nyc72s9K&TZy7NojQ?ym0|G9XqT`q5zbrhV>Es*&qlw?iI)LB;^LZk-"
    "d86(Dw1knZ0L~>5AIkkY_aMy{dj9_#y#fFq^$t^deq;pa1+sW*!Im7Y*n7B@@3#a!YU$pYhp-"
    ")sI+pb`kePB}fY!n9K1O@3PhO$XdT<HCqy%6&~y7;ZMaY<H4!d?v2316^D$ejUiIkuYnyw2y7j&t&^q)+>Ldca^}X^g1G$)9^Wre"
    "aJ~EQc^ls@xiwAZx_;=AC%`%PnqcCJ3P<7MnroF*J9&XE0A2TdkDtRM7qYi?+9wil2~%gEqV>`0wYHFdJ0u54A#g^4-"
    "B<ZL*2HnYNw;{HQrRV9{K(Cz|61FeWZeH`E8ID3N;{Bz;Z5frshE-b@58ezF&XaMhPZObC`!QCRF>Ov%1ZNDnlhIp-%v-"
    "SlJ(@9#NY`A@Gb^nw#*qpyNj1bKmOhp>0cV-rNI7d0c$1@MTl=Z?;UM!Y`OP<@L&^>g_d-"
    "ZSN{h|05|Aw4Qu~2#PWOQ!Fyko1!kp=<np_^RE^h^f}Wywgeo#uJv#v--P?espflz6LdA{e52xtOG|l_nyl6%aCD!kZZu6|U$(sn"
    "Zq)NAU^3d%ht?m3G0AsVLDUR9B(!mLEn=d1Ed=ZmVWtNcGCIavG8@B90nS`<ad@skA$T%3*Nf!4Pz_7;!&}wUNR2F~ngWW)$5qhj"
    "53tmvIuFiZnXBa3Sif<JnMo(ghwu}QCOxX|qjD4GukJeCyA!f3TPNG@_WwP^Jdxw3KzQzE@?|Ju>$y08l_^Y+GZL?U5N;e`#P^;O"
    "~vlhnx1CetL#~s%2@EhEXe^Hs?RG*nBci1x;2=G03>@<Lpf|)UIsCe51^{(-"
    "3UD})HuE+l*2)hG3M!o2pAH(6by+*BR|F`@%)6sB`yKbPXrGJbXb^8o2MkT*T&d_Dji-^fmhW$>VdudBo1RF-"
    "J!htB2CS{K>E0Ib!+xM_|-Bgi1Ka;O~Cf18iQbxe?8C2~4sjy7*`g<n<+hWKj|7>lajq~mA2Y;I2$begthW5VmWoVhs(sVK-"
    "G>e?OEE(3+x?>{{gad|l5BGm1-"
    ";I4r22C@q&$l=i2^QKXoJD3y@z}QP`^r~t87nZ*L~iIlPXq@_q;0<QH*B1(T<rVkLK3nbf~|ePw+Yg@-"
    "?>E_os{LKVd5p7b7<sO0pr!F1672bU*ixpRvNQ{vI^YK{O#@PC<MLtj3VxxN)?~NzH!fQKS9Jf*gTq!_u9JJ6un`}(f!S$+#pQ?I"
    "{oml7em=44neoB4=u1_?o|7oZHn}#u~rXkT<yBN`iaV=5jw`-"
    "uV0E}pip=<8=wOvs7iIY3?1Z(jW<%F<vZF*ts`(2N8Cf*q<Q8^4J^3E(^#hN`ZZI-"
    "L77qT1U93xZo0=3$>&Kk^0wxI5!m`ZDtiq6dfK7_48iehw^(0wwRDb@gEOAHqmF%HSCs}maCDJI4Y9Y!YDqDzIWX-"
    "R@KVXAnCC5fzt~}-@Fl-"
    "`s1L1|Qh#0Oa#4FWmN4$$_$*rhkE(HQS@LEFQ(qBrQq*Wv<mE}qJcOSUyrN~(r=>C<BNoBa2_kn+OX#BK9OiZwNJ5J#Mn*78ilwj"
    "N(~KRV`FvM4ZE8lWhH4K|r{95}wW?jZP(tIcq?nGPcyo$_nAn1UD)JEW-"
    "#b#)fdaVAu0I853(9=`gqpi)R<ADoNU<mGgOIZOC1z3mLTgqnkc4apT!C4K;hC7GBjDn9T;OM7qE&)oo7S0dyT5?u*BsE#c*DK;Z"
    "^pW5@+Y$PjdDnp-wUBtkg-NB+MLQvU<2?QyaB|w1oDmD(PweTm)yN`Mo7H)31jb6Drzj`Kfh}C$@DxPtt-"
    "6lb6$0!rW}JNzFl6aPs9<ZQo&M3yu(L#B~^-"
    "Av~O?tqw*{F_$yc=T6@0>aZd~P+(Ni$?x05U%b4EsEAZBi0{%!AUD;~;jH8Z4SB|c(oihYsj#f4MEcX%>rBMsUe&SXhqV{Y&%oj&"
    "q3xc>p1KRd9e_wVio>@qo_lMg&k~f9^KfYMya!~AM0XnDl#imay6YlawMMx(}Y3r@)s>l2z(Wf8xqa%Qv*)~R7C)9ZZuJm?)^V>v"
    "|1+Vp5M%Ri93Ejgk^`qbaJ2M!ES=o>r#Fm+Yukh?Ai<9>$%roIDOIH_8t1Y>ZF#O%|ztQn<mL45nsy@VSa{vCG|86Uou_)P51_xX"
    "!K6DMH3=^aj+G}`RByC>Jt*Jbmlp+V5vfDlwj5l#+GP8(L3itR*y+WeUS@f&VT}U-"
    "%e6ux$)<b(IEDYAK9cF4%B_0gBOoWyX<8j)wCxoDZ;c_-"
    "!lWSAU{<hQsOE888aTDT^F4%m+lDR5=kB<&_x9~b=g|`7d8Du|UYIf3EmH@O=+srT12SU7YoZrVdCgADssA_8GgA%VRi5F?Y<dtK"
    "Q->cvHbIb|^`9+~Rh$6_C)QwUN)(Kg&!;vKzVRHRA$;%*N^$Hz*i1P4jA7at%oGL3MPb|d?6J{xg%_4~Pyb`@132-"
    "mv+x5+$OvK{Vv&=^(Z2hgm{`aQ{IhXOS=H;}!@>O)uk05t7m4A2X9})@=x9>?#V$GH@;brXno;=3O*~ECWJSvK&XCPsw31LE6(QO"
    "?Xu0RTC^)joe3;S21Ga{sFR;~AwA^y%O;sI&QSCZ+nv-RWmghlmW8^+sobwrWbte_u*`KQGv<ZHaF*dl(WDi$0u7TA=JJ;C<ai$C"
    "Bx@74Jdcv2j``Sb?5Vm1kSjW)aNe|&Gioh6FS-"
    "<#PNoF6F*8!Lbw>cc+VjE{7l+t6mD_c|vP*JBRKC`a%Ud&SuxC0S}$BwZ@0>9^s^kD%|LMk+ASQB4!(_P29{QI!SZdvqWlx0pQwL"
    "wb;zsZR_1x|C~e?TZ;bV2%07L;Q`qtSgd)u_oN^oV*MY6~!C1N<CeUPL`aj(r*3&h19*Cf2y~>r11X}z(zN^ev@F4Zl<OdW8JLeZ"
    "8c+}XWPq*M23D{;YvXt&W2~p#9PfD?`gll&?s-x-PWTU&o@aB&TPo$2@6a%p4)IX!<Or3G|>kt?K(sS6IG+(hy;7?Wi1?cQRu?;w"
    "t+;;3QGIeMF5A9H?5_`EpTY#1Mhcd_@(){&G`3id;`?4aaC_^t?JlOn!E%M1EVy6lG=(t9n0ovKe+>4lMms(4w#&RP<D2R&)4vz$"
    "SmJr+05a)77kX40S4cvl^rs&hy+4=pWx6ML?!O?nJtb1DrJ-+Vsb}M(XmUWq~_fXnP-LyXIbIVH<#+j`kKXg9zlS6jrNwlZoXK05"
    "SAVF#yWm7YXb@oElFDv6@BpvQE0z)sld=Wm4mb0NAJV=rj|_@I8|F|UQfkYF*weVglV2k$s9rZ-Yao)++;$SNS@87;6+)3Az5Y`O"
    "V(sBMp%0`DD{kk0yNrV^L8t!yqR3Kbs99KAT8vz%p~L1V{4NN=F<E5pyQs;e%0Ujk<<iL#qxe+S}I5owk+!MBMN1nVZhYsgcG7-"
    "&&=9qQIuqB@m@e$)tb8Jtxss0zi<;OBAdu;u9m6Pw}7MW*x>A!r{+{_W0Mh<KTR;$l1<{~G>EIL0SVcb32*BNT5zXFKQqYOdfFX>"
    "m)aVJym$dek~;)y+Sazhx%TjDgh{i;m-Ht{?7ZS(@02*r-"
    "IgCwW%O~akDNf9mB&0+R{#{y)FgVN&C{nF9~zF=GR>D}?dBrhDWs}xgB4>;Kqp-"
    "~{Lw{Nq1c={<*=tY^dw*pUqfVdo&Ao!Btcl~He4)?*t(U2YM-HH@{hMHDTx^B?%kryzQUrJ^hl!wuo%Q|d&Z`!9Wul4{7kF5W)T="
    "+@#q9Tw^5!vM)Xwuo}Nxpr9+1kAiplOl@fl90_q~TFLEsJoyiwn@b7;%nqBcm@}kvgt*vl|H%rjXCDc~)C_Fq@PkT?aRfMuQE}xR"
    "@il7%tU$)cIBGyd}tsZ^eyX8kge!GW~F;Ii5JzjY55^LUOWyd5W08C{#xnB30T=_HkZP#R`-"
    "T@$?LX5l8{sYom6y)})A6B)tO}MptbtnoqDnW4Ab!6sOMlc~y*#}lr>D7q9eEoDwp1aT*-"
    "{gZebUZ3=g9jVt>z(2qZX+PGXT*pUP9!)7QFxJIr8jH=jwhIHwgOlB9-`;BBY~4<xvQ5G|5^tij=`F&S5S@~+2Po%dBd;Rvm-"
    ";H#C8iq^Vt9PP}AHL%XqcD?cZ@rE*AUn6{X?S!8i$b)KM<%mYP!eszza$CJmJ}H=&30v&SU&JbCAwuIDvCDb<1k+>?9vN|RX&Vm_"
    "&gPxG}55#S;GW;~L|AH`jR<=*_jJxQLzm~^K!qdT0bU)8q3J8fsv*2G2b;g-"
    "9>ss}JEj2`G=vT3+RT*qc?=hh6_9Zfz!{UdY8!d|*st<D&GW~{l{(blr$nlMzEW2|5pC;RXtH#Xwr8XDm=mak|K0_aV@#}uJlUm~"
    "ntIyGd;tleuvIg5id-q1L8xd8J=2D5{aXO*y82K0yx6WcP&$G?n}KR#_cp=6!6&B}OB6A?H~9l{IP5>E@I!0|7Q$h#zA>vsdZv-K"
    "A22+z4~H9?z-L1MT${qH>KdNAO5mj?S+4>vPJ4Or9b(I{EtK`Gp(zT1><cZlUsC1KX|leos{fmsrwZ-"
    "L&jy*wk%Zkv@25t?}xejO^q1b*&i%DHE;z$Xh6%(a{yd^N#Pfs#2DB&}5ED&??G2ut@~d+Y)rH0sCkrn=phbZ;b)Ya6~}#Il%l4t"
    ";S=HqrhzbI?&@QXS$QCR=M{QfNu+QU<%Wd1uojo_O)<|K6{Tl#`M6m=%yY*GhKTva49*K+fG<DV}Gb#{#H>68N6x-"
    ";bK{tf9;pgZFTqY+|^G2G6P-KjIzO>r8zU=WPzGL%5C)AL$P&?6hJnO+u4-"
    "V&CRBuAV_UH21VoXz6m7tbYl*)==XIA#BBeBz)c)wgpQ}blVy1{k9Kp)XdW^(?8sL6=+yQHm99#Up}yoU;Ii3b@k0p{=%ddxc)J^"
    "A_5#qlp2JZ(lB6w$x)tA-"
    "1V@7GGWP^l!(55VbY05oP_!K%|)27u3(MY(<X8DRAVIBuzTc({|u08{|oHR)F1!)iY>Hy+1utR_+E|Bh~D^XZ`(eVY7Hhbx!pv(#"
    "W>Q5*RMnE$j@_5#buKCBz7{VV=Jsp$2WsCGsdo-vl!ivBuXEn7{^k;x(Kv_YP?USH_#9R{e3WqO6MgvjZx_Rw#1Doh#BMegfE?jn"
    "NSsr5n*H#ux19a1Wg1QUpqSK@4vrF0)l+&CK7<25bE2b{q?BiOAx;JH#NYEK{~lL+cFBk-^)r3Y98~96P8Iu3vIm*PB7k%-"
    "@$=EVGp#dUwFZjtrB%Q??hX>-csyf+#Z##qJjukfGXx_5rc6jLDv-"
    "$#Xu=zPqKCJX<i*kKu>xkugjwS?DUJ6Z6+SAv5eNL2^HtA+&{;>E$f;=d5hSM&4ChB8lBW9dmD{*r_@~<Q_0hF1S{zcZgx50gSx|"
    "$U{5g|_%$pAwp&Z7fnbDi?U#KMXZ<B+Ga9ER?`-=^g8__m1>jRxcI+4dDAG!_i&AFK>41Qke>$&L#Ip}-"
    "Uni0E2__&`dXu|;hHZoHY*3XOU$?jhk6!eWwPU9#ZX7#q;YmAj^}n^)w?IDz+s-r21&<^<`+2Yb8RM2;_UaNmxFinyW1B-"
    "kF|`yIUDwhiiGZ8P8U3I$BlDg!Ly)B4bT-"
    "X1O3fwmo}mmHe@ehH`;8)Z93F^lfM9Nq`b0fzwyPXa6y?>W;6<=bC6eGB9w%Ck))d|B=&%<#Q?TZrhf<sUd(`$~=JXF_LX88V@%("
    "*N+kgGnKmYmf|NO82U2#8_wur}3B=pc-2fd*U-"
    "JktE#lB=QVMYxMNy@H!hCa9hi8QrW^N!b!Mdw|#;19K`H$%+3&=ZIVUxB{G!GG;%gz)<pOa<Q=0Mvdqmrd^3!4!O?SgiLSn<J2H`"
    "7vwr3OvYr?5axn*&9X`6o$w-EHT}N@d|&OLDkWzCP#b$2fb#8(OaZOH5PwP1BnxMW$bMlNcGzr6-"
    "@_|bzFNBo{~T_IQ4xWlCt;{#j&~A_Ca{Om5GAUWOhOW%?qC}3wfh9Gq2z&0NXtVplO5}Gu1I6?GdeWmflf4%+N2=KVZMOL2JJE6b"
    ">u~w&j~MQy37@E^|U3NntT@4;i8KtqEK?;~HIiJkLZWCudDELL&gL0PrHUk2dq8Njw0h$xmB8-"
    "18d|4B4Q>pBemez8l3Tyr15w?AZ;4+tMG#V)IYHztxtsZ!tHC0q+$o`?#&i)3ot`e4Q$FMzGJHa(DI{e}i%Tt2|wLA$=SPw$<Xh3"
    "1&phN+p?(fpqk3>D+1d?uhAX{}w&6zL+_2q}PsjvzlLvAwE8Rl64_Hy1HbasO0^v=(v!r$Nj3B24(Pdc5u~3Cnymp7Nu<zbD{;FU"
    "i|G-**ytqaJqKV8Z#z#bw3uJVCFkG6JWpgiuuScA^TshpnhqowQ_>U?&yEpMrlv<y9IZdZZB#+<pwWq`+)rr^x6KBdB``}X(-"
    "U%N4Di2e<lLKxaU%qH}QIlvGRTbks2zhV_AhAjaefAyuM($9*gz-umAVY|NYPZ`JI<F5(8;5nN;GogY*aiKk+APE4tz;jjmdh#VD"
    "YBeOo+ww$59#+G(OxYg(|@lEq^LYo(A{Is{A(ihJL!a?Q}3XEAi}%H66@%D0vvAsg9AOsy@oDEL0;K_@H$V}cu&OB6q6!k+xUtuJ"
    "GIjf76?Gl)k^r`mIFM~}{y1RFD47!!`{+G(#zwJ*EoDWrCz6N>Ax7vgc3Cwj@!rxg`dD}{=?j5<fqVGP7n+{fJwi#Un|)2*MWW(^"
    "P)VZ+d__oD>S7l5mM=y1S{j$Cny;dMGKn8B3vZv&H<ZmRc&`e@ab<5_)46x)R_)WWt$u*hCgUJ(<K_@W6wThJccc-"
    "X;&B(vezod9eQq@A5auvIGM6HDhkKs93C;NOH4?xhPrQw269L+_Ae(HdqWzyA}7@jrcfQn%$P*}H?mV_f(6?_2H_SMZy-xN-u_-"
    "A}<{FJimV&uzLpm&qO5LX9f#(6a87+w(|V$>j<Wz|3<Rr6<6?E1l<4DWI^i0%8&;JGYL_{>}8Ur*_<@8dJ$6Sko}E^4sq`@AgCI#"
    "C3`2GF^*~Vd)F+%<mZ8&ePV`aDHW>yqbrj6RRjo?zaYKlbaT0i9HWx_u24@^<3(lbV)reBvxzU?KuWN9)SgBl(K|*XHU8_t<!;fH"
    "@8k*G6u4~`judf$2DPp<HE#T9|0zSSUzsavXh62WD}B!Mxs%QXJqSUC&p~u2d+W)aY~uxcPWD`;a}cd#ebGjMH@fxCPVugxwC0C&"
    "J;(@GSU_*=|pH9iGrzjN89e}P4QQfk;rsn;YF`$GEiNU%zU5qpYV8n{k<<w^))6B2lwwShrqd;`hQq=uuLcJyu%`}2zWcK&#q=68"
    "tl-1lSwRflqY!V$pho>pK|fGuy7|Ixq7E}Fpj*l!0oezITGC9HI(nzmm?X0wF`7PH`-"
    "SsENirTsX2eCauHClDtoaRW49|#5Xi)It+}ZkoodbFs%9b!-d|hE|Im7UrX3w4wG%*IQxF&eF7ALWbL&p1#hbg6a<-"
    "1s$F*l`;WO_ka>PKzjlV}<Wavf}nckzi7`Z+Cd{W!Wu;r-"
    "(BtxAhT|3)%`#XlT3&iB(nKI5*0mvzS|ApAJ6JdbCVk3R2$=&%B#>97#P^@-"
    "Gi`H1E)?$^+31Qyqypu=VC^qQuOau{_?=ngJ&I$4GXF7wek5F0Rm4bPI%GRewCu^CFNYnoF^~_&-"
    "M?)<kpdLUAS#OWlrs@Gf?<g^$e6Jl*4gM9pnFpjjB-9hwNle*ytEm-"
    "$30FymQtP3G{v5#<J;`9NzuB(Lt!ycI4cwWD;|(Bpoxaaf!ma946G}YNkR#$elGjfvb7ct@ATY4yh!O*e-LpY#z3p41UQ5-"
    ")k9JysKyETFDwx_iZy{&jx!F5`F}W~!4h}KO+!D9)0`gfW*1YH0A(sHDY5ty`n(IZQk9+UV#!bP$@v~iEm)C^BuSF@?SOA9ctxmm"
    "3e|MwXv&lo$IQAO1Nz>@f6}kzGedmxjq22;z3>c!dvM^PSAdF+&juuS4SyVhnj8=_xH^ct@>cLxK#x+$$`ZPI`53lBumbkC2?em?"
    "z4+H`JAzo8+>#<ssFqt%#rH3zNLw~KUi~_3m6w>B<v(kMUYn>Y_VK<Zjor9#F5y$fc-jSh~jwlSAaASIyRM)7rG2zp2kM}jO#4Sj"
    "qX<C;OtUd-"
    "C2rb8Qd?|A%Zc*v2bLpB~P!jp9_U5O=IC};O19Dgscm3F794_wqT)zx#RZ&Ln?>)j;89(GRM!su1@x$l3;tOfa>Jzc`nF4ZRt+fg"
    "O6>D+{%Q?mxaOUn`dR>?Zadi{t`PP<BZAOCT6SZ@-@PCJbIP-rUJpfE*Hxao{{A$W6dYg?lO9-"
    "TLzx8XJD`cNnFSi!usI!Eu^sZD3QT9@DMYh(jTzU<K5YvD$`hK--%@mrb;7f)Jf|w`HPH`Qcu*Sl~!d^b6<ZF-"
    "<9t`~6Gg#vSWf^_@<Lat<(C^Skf6toXQtKPc98?YoWZ#;yy~Ica4D3W_6B=ZoHr|rNvD=5EXM|*+0)<K?Ovk+3NkC+f>Q`T-"
    ";Drv0bYI7X^d$K3p~PI@(MK>=Mj^U1n=q(Bhtrev$Pp)QL9i1xtz*Pr<3*#w^{FnC;cmoeTG(m3b)0}yWlVeaJXGsuY`iAUvN4{!"
    "EFW(*F5xl^n=_g{CKnQp+Ff`JO+5WPd*3`ZQ%4~6s-+T3Wo{*)`c4S$J8bjIu6qfpwJ!HO_AX-"
    ")IbGcuL&}g{tEB3g>ye=O@MO>T%Z%{jSR<}##@3{8jmWP#Vip^}{2QGXqYOz*f8l3D06Mk=MGQL*OK6Seu`@wLi7*|p|GwMg+)#4"
    "v89IB8dKQ^n^f={tYWD7;4NDUYyltH)cqcaMSQ2k}R-qmmsndYaXD&^RtRKgXtKq5_NFG6aq!(#tH(bY)CsI`4-k6@%&<{TI%`?-"
    "W52k&K?c3g9R%+UYB}vc$$@gart}piVN{;6B(LS$a4SCkas_e7Cv01_<U5DXG@#8Vd)Us8<HZBopT5tB_mLN=>eUje(8C{@)Tp2X"
    "8bMjTqtcAus8d}$hxwEzLGgk$0?vc?fLFxeHGYoOYuvE3=WXW?tka@*<I!2%~gZ6~@K4084744PIl4flZM6VD)3cc51;%~*KoB_u"
    "75*0-3${Lm$J#rUL1H)Q9@$}+PWe;+?FK-"
    ")g&!BjgWW>!LB~D!`+a(O05mU#BXm`CCqQ3*1F_eYWGKrYv!)ja!&NCg}W*Vcp<R5;M%hJ+^Bjj)kD~S1;P7tK9JWE6#!HL<2-"
    "3B*JBE>u8F;9o`mDzGYc~?BhRP1Y05V=0r3cW15R2?&lSaSv*s)m0blv(0T*s^*MYs*c~p`ts7=>O6Jok8Pl$4F+M{OX*Da*8tCM"
    "L&X<R-"
    "V~Tl~m?2ir~yOEv_RI?RLA8o&F5c?M!&7s`Yheb>p9+)gCqa#JWB0iDTI}Yg;Gz*UU&h)spw%i!eO0Hef@Y0yq2zn>Qo<(Gij~UX"
    "_J+mg?FiM=_M-qV#+12zbbbNu@}3`p{Hi+97dYz9Xeb_`5Zx$zWrbO@5A~GLH3Fg5K{?ca|rQddR~TN0Lt5Yu)F@VWO6lWY@>2XB"
    "ExLQg$meL{zD1=ICc9q1$JmC#Ju-"
    "3Wr4i9lQ1t&+~*toIqX_krAl(YQm|{2FUF(ad{4<BRhUf{drH<kV$S93q;`Fyi<@$sQ4EhJrXjW*q9r;VxaML)*FKKTaqx^gY%@s"
    "H~o!St{YAkow4Iu-#mAo)pN8#^br|tB*?qG0F16&X~Z_QcP7?*W4JPr>^T;z_r-fp-Ee=D_(vR&LO7Wq9rDhU!JCW$-"
    "=BmATV3%~mQ9sED>fIu&mq}$=^38?&$qawNFBVfA(IosyPSXd0aGswEJ#L;VBO8h)`qPq4UIkCv$YRxGHvd#EN<>XV$vZ4B}Z_H`"
    "t=&PKzf4uW%f>?Sg-KVjMI(b;@DYJ{jP<1TS^A1tx^XnwS0Knc$@L2nUB6uVWTzG7)n(LLmxyW?_0>^v<=mC@y+9~y^*8#OJcBa="
    "5K!Xma(Xv^~~<A{TdPgG$c8-"
    "72EH*o|+2n;P$Y*;_SyfE2<!w+gpFV5m(bI#g%t82>{J3@h~dpt|CI0qY8X8oG#%f8Mn!P7$ZKTE$kf94ad#ef)Try%$_5@m=ULz"
    ")EiEQ)5{-"
    "=BpOzd2)A9251$%qhdG_|I9?CiAtJn^PxIWZdv=c&hP`#dr6B6jNxsA3&e?KAOqHr<+LSAi=2#HrLetx(ZOUMIrp;6s4x5A%8<KP"
    "u7L9N1xGdO8m#3w)XFG{J-cGMI<}O>)6C-"
    "ln2ZsBXF%`vi`}|g3zWDl*LeTXbEkoG;YT5P<zPJh)O#)^rE0!wikfib}b{gj%nXp$J8xrYkX6lJr&9t#H@D{}7U)96`N}Jc69OA"
    "=ER<k-GI?nA02E~P7N6=C#>K7exg5;0!=l*`nBS`B!O5<nOC)+XbDSry!w+LojihI}k0WVRI8)SU)8`1!ga01-UUNzd-"
    "qLNtSZxd6=K9!NwHP-WxckX1^q50P(49<QOUFtdh{1^e5sq%;T5~HxNCWfVfPH-"
    "z;^akZ$?zp^P@#pUsgLscM*q+$35ozhU>)KeOg;LbJ4_j2h>O~jp;yLQD+^7PR9}~a#KMfw!b>h_CSHT$8e0LfW$bnH9dr#y?Zc*"
    "}dS!<??tMPAnyg7v#$r1ebS!)reyp4HfYu2F*^L^`OE&aFqiyvmA0)f&uY}2+o+kU@6BDStEFuTOU3ZdzK$9)m<_lZzS_jL~z#{k"
    "XWDX*+(xZ3F}Sm@uodrZS-8<CRDK)$=HXzEmT>p-(JkTkA!ejA!iQJauI_p&p*Q(n#a-X&wB#w}JPS8$IwUDZLV`OiK12-"
    "=!%#<@XxB{B{u+47UMVcA}c->WZt6Un!S*ShwCGszvln^4Wx9=~|QeYjE-"
    "*Zv}LsR77gW0;70jghcs6E?d1@1x|>p#cpzv@b~Wege!&P#wrL^}A-"
    "vz@#YYDmgWciI5Pqk)AqSrD?s`!$+%LDBfh;`zp@CwY{O}D((p8j(KEpa{Ld2oSwhm2~H$E*3Hy9ZBOcOW>DK?nHXpO-ca-Ujjps"
    "04qK+!^0_V=Fylzyl8JFfxY(0=mF~tZeN0}NLg~mw$nMc&F1_8MhI+o5$gw10)iQAT@<A#5WSCu{owr|;Ips)d^ZWKLjshmhpqbX"
    "ZJ$I_Re#wTgW{L<hVNe|KF4qK;!3rNC;KzhIf@OS^1Q%-"
    "Id&czM<)}~n)PjRMzl_7E1&qXs33l8fnV#TdAFRnIwP}CfZ_*zLk4VR=LBAsDHiK*b1u?D(%WHk(^6FQI+(fTQpS5>fBZUz+;Q%M"
    "9NqoWjEMLqL>sLwo3h|v0(N!jzGsR|fecF%4p%VhE!HU`rH6a9lFZ0?rVRC)nF1g(?TY7ElnSi15-"
    "=I;uJHkO;mNgzDPB$G&a$fh~lcs*AcXn``Iq+++U_h`-!P6VJb%^`iH}ZbLwn4CDn4;QQ+L^UWW;qgEwtTY35dE@?SQ-"
    "Z)4E{;>IqZ7<CcZsKDacc}44tQ^o|aY{_0cy4-iq!DnlLfO8K+E5Is-VKrQ=9OuM<DN-"
    "%)!h)S9FF8Wt8FQ`7ZdXEy2yhpu5K3)o&|c6K*{2<URwXjnAzHlMOo?{n>Nvm{B}N&UUwIX-"
    "0Cf*!=EP$hP5VtV2`DC2W1kp!%G(;DffC|hy5e6Q<^Lb8ZYVfkLgE^a)*osIpLn__F3t&`{ilqVfL8qI9|dsuJ2iC?L){gT-"
    "#ch=D<viQFbz<}5B9(w^_Vmy@_0^%^s*DdtT%!|*@)$8y{Y~9C8STcOuk|{-"
    "cRhBD?bc=Kew(G>4svfwWEL_yp<D?F)IxFCHrY2jmI-{1iB<(Uil#(EC_P{@LF9%v6gm-I84-"
    "Av<0qpa9CMJbm>j^lKJMo<OP!wTs8b*TNv%s(031U=k=ZN{RR1@y!Oh3Da0j|k@bsOOSnu5!A-"
    "lXmswT{#4N#v_Ot_L|euA<&O`u(YzSN&j))lypEhzPzeHR9EicuG<lXq%&E9%w<8*$}Lqn86^7q7ooj>zc`Qtt!np)O-"
    "7+!zV5Yq4=oKlter_1aUjw?sk&%XUV-_E*5BA^?A^HDOB&`JpZ1L<bIq9q1q;BstN&g8#h=@WeLQxyH7Oh`sqPvf2d%uMojN9QOm"
    "&eAD_IjNFc1H?~0J;I|!Thjg6pJZ_KZVU3<J3U1%C>vsh$6GiWkw?8HU#C5;u?HxOU5;oDY2V#M`LI}_>UJq<X^xsl0Z5<f6UHWj"
    "Hg?$ufuc~Nmq*SGQvt8(4q?l=-q-&}yzupAvhB>`49tkFBk1r}MeYoqb^uh~nNu2m2CM9q&F+(SG!#v8+-"
    "VsmTD+LJH}F*+K!(zPE+?k2!l-"
    "lZpd$R(LDK@*+r<Fv@a);j(1I{a9dE8jG1mvP{k1t)YM^Vb3m$ohgf5nq#kMl14Ueo2fxpkCjR>-"
    "drcH$z1s=xU}=k=yaK!qUOEKdIJwGvv9Tt3iD3prPm=gZ&J7b8Irv7kV#$wRy#sJyGNBk(sZb(93|pXRA{hcz1XJaUrRUdK6)|sB"
    "3T_9)meJo{+bZAvh_T5Cv&#D-"
    "CS%oYPWoNmDj8Sv&sQgRhpLvPWR;ghAg$nbtK?J#D7xkKtO8aI~Z>&!Uw}ok%TGivu?1m5EC*0DhQOfWNMJD8uL`b@h60MXNR7UM"
    ";|E)z<1hh(Qyi7+vZyp^_?H?;nRxlOUuHar>)7G>9E%&Q%2axjkPA4u~d1TZ&{nZ6hKj^VRC`T-"
    "7Mge2RL3+_Xgi{Dsj=C4Z);6I|EKK#0hrn`5KR22wnBTN@?z99IGq?u0s3dly*v7W3d$L>=J(#^k^Az?!k}(@*K<xB(O?!vYub;E"
    "29cw$xt#igNrkW*8Lz8oIV3V|8_mJJHCCj*;JbE6S_~)97{yK<6Ggno+QkU7Q`I#xz;8q4Hb4b$a^2E|zmWcq+Y;_{4*JV4V!M44"
    "grAY7sP|WvU6H1kt)r)#?yQ)Yp#!d!00|U{bNoBiOE}x!~u3%SudH#0H9A1Lrq1u*N69NptlsO)viY2ZFmTtL}vAq~#I%V|;;h*|"
    "JoHZT7BG!Yja?vG=gP6il$sji~mK-?4<}G(2a0*I{d{9^5-4Pv+OkQ>t{Rjz_w}9k<x>^vZxhA<1B3q+1PX4H`F&OQb7<-"
    "?5Xmz5Yfrbt~=d5$uKx(Vyw*ZY}5)$T@#z4>dT+e_uAr&RU&bny2v1NwMGF(}T{Q$p~lkE+MdVN&(Q$$l;n9<h@Jz%C;dQ1qzsEI"
    "kBmCrC+fF-t;6u%+ZV1sdzb9m^g^^ht+0nzeiT`tW$kiN$zPqzoLqrECG!-2J08Uex=|{Al%sthTr$S`D$#B*3!^fR&%r%#A5Vz-"
    "+G5_#lWk_YP7;$xzO_GTX>)?^l9!+#eM&t@(XDIK1GGnt7*PU(ojeGkXOAVq^Hy^tw_{NSnjf7n;rJ1o=nK(TMakx;x*@X!VCsa="
    "DbMAHZIpBXhsGu_lET0ni;eXo=?K4`su#V7W)hUQ8G@qsg?sQ^;XgC`;&8~8)t|dX`Ko812XTw@J?<IvQR=`%^9yT39PXo7r$<)Y"
    "SK;_@y}mYnSg6{m?CPUW-hDLo9WC+jcYDnx{U0ukf8VA5W91@@{-"
    "WWaOPKs95Iu{5y?#7YgoWr;u_d#d!$TN6!|GJC9_`pVG@kMbxU=V)4|hh+u*)@JF!HGePzTC$WbQGLx80SW21m=1hzIt-flC%^=t"
    "{Ju|kl;H{x4u@{i<AdZ3=kFy8V+aY?O7Wj;Qhz9ehR{<G!}XhY|os2~HKu3Aq_IAgY(Yq6&iXL{o$<kYj79bD^|`YGI&sn)JNyck"
    "^xzueJ6Q-=XaspFxZ-5s%Ff8$Y@{idn<lV~1`nZ6W5b%&6A%yT>9iFZ@O!nD-qF?HXQFi^wHfp+UV_}}$)+W_edL0L=nl&uGelhQ"
    "v<W>h<`ejSPcyFi=zZ(i37X^GQJj9TQxiw7)%TlNlio7?{o4w}LKOcEpJg0muzV_d4jObezSfPSoS;jAhc(R?-"
    "5ryZVT+(c%2Mch+0=_=CiPCc3#ZSP)K5&5dF#7Fc!sJE8_3uSW*$dAGMuw*RNOXB8EVE~DBY1$f*2_vs9p2}(cFp03k#@*2+lvfz"
    "(?PWb&=Nf#s9LP@s$tF|QiddVKJdL+t7{4M<fbN|DT3+L_i$oLYR|$-)#MWD~zILtps6DTE`L?hW=O-"
    "lNFzHxTs2^v99F;_jE4RVQZ}J3FsfFEE!X73<qZb;l&iUfA)57seH7FCxlS1EZSRVNT_eAI|j*r**`ZEOU4<vMsjhwoKaG!7_6sL"
    "0($7?VWR{v^)n$(zAy*zGTRT!u%x7Vxk=>$uIq!1c-qTl7iIC096!tI02UcL9CDJ)Cg@#Oo_^{1PHZ+5$(<Rse50`#eR&-"
    "Ruxw@Q+}gjHf}#?~^Q+5JxI{%n;C-G{$5vjV~bWGf<xG!WSqy3pTqIa)F_j}Z8t{xIks4(T#UcQa-"
    "vZNieJ#P3&G3PKaLf@L-{#WU*FD0H5%g8uw=981dPza-"
    "osj!$SiTa={@x>;u^9@eDk<^GJwIsk*;IrB^ajg!<4)QaUS#Z@eG6z;BrzH|&|b|IwM?DfvZC?Jv4JV<TY(y;_OVE557Cb<-hEX-"
    "ilFXkI@id|WR4KPj|BXcmtWnHR#CsMM`p#NsimE|Qh2Ay6VNlOwusT-#(@*g(Jk-"
    "yqY5S7XrFbO%uWFq0#=bXBtLfpPwBhbt^A2T?w>X?&`#uPcd(`eoRt%8KNcI!`!oi5p-cANJ;iQ%uC-cJT{Sl0d^p1d9ks}7m+nb"
    "5L2clVxAmm}lO`iSdi;)&1X5J~jto|FpkIW+uoPq|t8g<Zl0jaTaE!C!nKfW>NO%8phx8W><s#Nbqdb5v+wtctw`*T0t2Q1&FFKZ"
    "f&bY#7mIxNt}N4svqv6Qp48vx-lL^9r2cY)QT@F*I1^!_^_(uaVJPDQ(%WRF@Pryxc6t#pxk)entJNjcHPWnZub3YO=?-"
    "Gbit?N7$I2VdC3pShyI{2cFy?sAmU{Ds%n@b-y%6U_Ry;EoSw`^BBgKGw2i_*m~6k-"
    "KhLEDc#QpHm%C#&cZ4F)wNWwXAs?L=k1FM&9)S6n%B4u7s)^dG!3x1wbXkC0Bd~F=P-QCr(&JIvcH~OzlwY#)xSI8*qSvc(=)-G-"
    "7{h%5;c+(77}kY5dzb(JUhyq;}dbN+m<m;H+zSv1n7r!u{T3Tkl7_0PjpcjYG061!Kk!JFjKhGmQ~4Bgpn;zsOx(IdBci4lE~#)D"
    "(C1PLF<)R?GCNRxyZCilte=j4U#U`!WzMD)8MJpHZ&xUQ?JN4&!XVN;4lAV9H(!lzH%XZ6D(gAt%pjWw5rJjJi^DjuE~g&@$wkwI"
    "CP6F3syoqnBDzTJeW1i;p@)!=@?hEoix*%+11ll8H&e5Av>CSXhL?&KadsP`K_AyjrKs;=HN6-"
    "<M(zCOzzc^3_xIvQRtmD%??r34>bZxcD87+apJ09H`mwr2+dm7dWzr3jEo;ZM5^DWW2fw~CA{9f-"
    "FkLM8CiQG&fZbVAo0C1zO`s|Pgs{7y?%T?*z8x!9X0Gr^tni82vYCCU-"
    "+<bdjzfa3(57E<pKrmOsV+b_5OGD*e#g#l<*g1`Z{@vT;VnlJhbaENl(t#C|n&}gHV0b)t_f9BeUs|z}CdzB*S-iC0g#t-"
    "q+RL9%>ub_>cC<FHF1(Lyq0Om3mEOx9ED`_#LN2Te5a2*8SRnKMDT3B0$=<@^r7!N&E53gm`qje+9yC+%z8Q{A#Y`hYz$Si|P-"
    "hVZPdC8NG>~(w2(odZsQ8FbMKxZDBma&YC}V1R~ti_6>j0(Ts*#fA>SQcO-pm*$bDVFB0*fj9H(n9kCvXEMCDES~`brzvf`w#ym*"
    "3wTEg;Z9IxAt2>5TPygLPd)_$TbO?WpfAz*%J;+FT$kV2|ILi=aeLcu#YkT(5Cmbdcn|7nE_zFRa^gq)MkaznqNL+hM&H^2iS~#s"
    ";hCIT8nEnhZ`XunRJn1r)ISw=WcuQktgK)OS)!H%Z_;>B|hFo4*Wd+ZYJ8^jtHbjE=ga+DAptu?)ncdI#w7J?8pEk|NbiT?EgUqm"
    "S9w)#+*RNd%!FtOjK(%H>bA}k+<~)3oQ$WPtvFIEddop+u8`9{Q;B}Bc*WjP~eU?EXuky+w`<Y%Mx+j<Ext4}v4f(|tW;NBHu;lp"
    "%TNy3$DPtfOweb)t^)?hzoH07z8!7u~28vxXNCf~*&*LJtHB|@stYOTO7_-"
    "M7$LPBQ99LZDNOx?$p59a*()=Xkf)>p_bW6A()>B{-"
    ")uY<Na3>Rk^_hGGRp3j&etsH*4%F;dS<@Hxrkb)#owiJ1p;FLT@T~>dG6=CVfj$cT3IXf>%|MU?F<~3qY^#wgRlQ2zyv(tWq8|$#"
    "oeb|J?NC=_<5?w$ik?!;r+JJOc(Zaa{yTaKwT7I%#2uC_r+TE^+1Biam~|F-qD&&F@f-"
    "F@oPVU9dz_YAt9&>rSOe`NY9MhYQPjJk^Qi}X6;H=%!I!sB2L12xcZcnn$6PIj(|KUz)2nMCs@Gppi)=*SZ9wU)$1-"
    "6|?4RwEeG*Tup^!XJJ$wn<`n}$peZtq^EJ=wWMVf7V$8@AuzOUB&m7vUD=BKbHS!P@9E&kQjTe3ZCs-PDV)tAzClq2@nwzmc^(@d"
    "O8=mci9(?^vT;z;u3UrzB^MsD_{qot(I{f(TH>5jOlXK%l55vxm1PZ<H0`2NU_JJh9ae?vNsaiJ^oV`GNCZ(Z57JTfqbR+1mwts)"
    "*YUu;I`jdtZ=K8><WOwY0gQ>IZ_IXkye#&8QnU=$>0^M&ml4m8O(>A|fnq3yiF>mn2SrwEq}>nTGRh`OhWf!|8hWORZMSfFfM$Ya"
    "}iy+tOuDE77quD2oj{ntXce%4_N@_$zF8WZ)L!v^1`=jMy=J**lI=5;B}-"
    "Zg7&6L}yq2Xcefyqs@pRQ8a5qI`~zy;Cjv9Q5aw?fY?#5E1SrEMV{Y#6%;Ho@&%fLhjqm0coIQIBU|h+qErL%gZJ|*|Em$jt<B7X"
    "*V>I3dPD5Oku}v(1&kG1%c`}Ha9?D^m$)<gN06j-"
    "5`KW@2hTPn%Y8@M;;$TK?NFH<QprLHb8?6Xuk(YH$%#tA)=y*qNfzpvXOq?Rt^ImPiL78a9P|qSwIeniSw}x13uj$&xyaW|0Lyun"
    "Z!D9>XnT;wt>Jnj{A$K;0&bT6oz+ohZosaNGWeY;iWs9%o%I;fWe$QF?9^}?V2{<;T#s_bxtPu6~aQ4c1rOc%}BUs4@2hIjxWkf)"
    ")*^ul+2~z+Y>HIoW}J2p%O%m(`p$bwYA#<<B@1`se6Y@q?ZUAZk2{paBEQxus;+FpVd4<HkN3oOsiGsFd6CJ)bNuIwnzVlx45++)"
    "n@yNQ!cCbk042kNa?@;aZWk~!}BFF@VzE4fK++=|8?ElmL1n|rQvtMHD&KSXYv{PqDS9rU=)d>1PcV%07SLhL+{@C&4{&uwrR-"
    "+RH0C{YiH(Ku_EUDr%CRPBQn5VQcjlOW=G_fJDFrXHI4c)H=CzMGacV?go-"
    "|_)rn;Cre4*ZBSl>kTkdkV8fc22Sb@5QMOuMJDyf4x?-"
    "?h)7$yRWQXve+)8_QaLG6_FnEMvm;F+?tbJ4dFrMv)3%IKrX^XwbsZjf6o%8>}B??k%QXB$<;POCzX_83`#u_|j?o#6(T4h>exV;"
    "9a;|ALY!9+9og&bmFBHEf_TOgpi6XrYj>*|hc3KKkKXTF6(zW=Ag~?OSY5$N!Nou_lC6kCqCh*GR(5l`7sjc_gdz-Epw)SJP+66E"
    "B{dd)H4ZZ^T2Y`>;8}HX3uQIrX3;dw9=XlJ8nhQ)K7&j~`KT7Z8-"
    "5z5vu|c&HMh0dVZKwxX6eII{i()|43;Sui$zyu+!<*_((V&DXgZ?D)=TrBM2JCDB&XJ?_CgBiG#A9)P{skSoIYI@#^o+?jm^iqf7"
    "G+mr-*AYy~gp?nKh9^v<{d~0ay2sMRp<)61Kkf`CvbZVQ&cuo_-5l_7A=s&?X3TZf!^y`Gv#e|XZ!J&bm-"
    "R;7n+`@?SEmd>EI_*G%^E6O`Y1~e@3`+c7)Ah$<S5&@rK-U%<FP}RO-BjHPNxRIiVHgaw(Uj^@P}V*IU%CBRjPFXyhSqEu92H~WJ"
    "?x#~$8m;J6#7|xvQw=Q$<Ij+c#PC9exLOIp)Zdf^f)!cwxNf2pnP13K}le5b%$@|m62S`;V?olokd9^bR~~Y!SL*!^;<K4HeZ8&?"
    "fCy?FJWbIVd;+!zTf@*X7CYszq$MUu;wW`6juZGO&Kw%N|{ra#yPLWjKrs8+N#-"
    "#ZPb6o$hX%7LoK!65p8MdHi_R)Jk;aryI13%y-"
    "1U3+&yY}OhrxACWR%YFyE4J;piDX)7PFLkm%2PauA8XMm<sN(UFCUGR;nAzO|_`XW(1+4|ztS1!HFF5w$}fw-"
    "SMJ>dO;5H>O~Ap66N(>1mvgZ5+Ha`kz|N!}8AZ5b+ke&C#(TvlC&^BGY56j7u6wQiSeH-UH2(+d0;{$8dhb!LXNbj%`iK*0hY3<c"
    "ukC`35D}v>=u}y2tW55w;V_$%NW^?j=M5(7^$0FtgGRq4E&Qc+F$PAbddKqIA8E!LZlAPyj65d@KfizBa6Qhp_tTdy;hxFCN<vC-"
    "XeW=U%U|QcwCw)_*YzOfAFPld96-"
    "#3DA*V;_OOc%Z76>O$nLyt3&xP6u<6A6b+8iG1ZvOVT^D^itglXp?(!*K*(LWGU{J*AH}Z9v#LjK2vnHGnkWM!_>SmBv05^DI@0S"
    "&?DK=fo#IT5e<Y9(C}_&$4Z{-n#}pGhA2vc40XV74{6TXbD-n*kD7*3=!~vi`Z&Tw<Li&4R?$uDowrRYWDnee=Ih6s`B@YRwl-"
    ";SOX`+Axltp&V}vb5lo9~Q)N?y+sw+@=I*NV7eKGI}vC%KCspvx}Z;WudC^dFlgSB6fx*a}_egLR|&IF4eYo{2{#mLCKh$HHf>4="
    "{h>j1a*A#V>iWG&Kf8G-Pr9c@u=4e!oz4qYu&;;EMT$qaO;CZ%CzDwBCo+|){jzhhWRDyMQf8aB~<LTfVPMq-"
    "@SVZEKMm@Dj&M{QblcBbi6l<lpm+!B$oQ+=K_GdQAeQF`2Wbq!chbn(f^eFn%nAt}+eFIk6r2uBZrIDS8$TX9((+bkIIi%r#tjNx"
    "$c_D=cPS0^+nuedYfx9%HtNkFjq(AI6#q&r`k4LM*umgkN72tv-pajwS`iz0C+WHXIC@Rv&k_Dx%dfzynv5r{x8ls6aX%RnFjfg@"
    "#C*qQ28KYA^}a(1XK3RpGD)h(6G)R9!R4?6d5PqqQZMH@ZcpmU6c39fG}r@l7n?)g1)o}I#Io9)`tP#a}JQ*o9Yx9&}(rrowBm0c"
    "XKBS#{!B3Ghj6n70Vw0CEPuVZW#1Gw-;kVQb~b3T<BG=wtq9C?9GMfbtg0O%MQO>W5hb=NSg<3jZdt)<FlJ-X*JyGa8kmTX5VUm-"
    "_U=dBNhkaK^SdQr3WPqI{;!MZr(v~S10nEBKZk7Ctk8U)C9c$7o*l~b&wjV&n?C1?r}!==!}T+dM#m~E^g85GMstkV!?(l$~kFs+"
    "96y&9`@$t|zm-(-_%UV)e$ReE3WPRYAs>8>AcHY9Rtjeo02|N5Xh1A)Ac1&3G1;F1vc9Pd${Iv(3RE~RDBf-T9sz53WY-"
    ")~BSA^<ZhQ(d7e8tR08>v@+A9)ZLF?g%ib7hg4a_~*Qtt@CtI){r8X;zmM{XxC?UJ0b>RUU`H~CRM($a&xj^!N@Af;fcmts}+(I>"
    "v9_;QYUD)jE~z0>)@V>iXyN>o64rc95mu<D7ug25FSxj)*Bg1e^SW3)F$Wv^lHFTSPU^zR&2v1s(>+>rF48KApLcR)UDAzqLEN+{"
    "7`ZhVgbixZSFDlCk*GaQXzMSARl{R3*;ZeNT$P?fG|C^nid>^8YUig-"
    "G*{q_tlPUJatU{lO44mk(5piO9L7*wL|@2v91akQA4+v^JdP<p#PBIL}vA6<XpwSEh1REuysqQtcFbyb5?L7s`GNDz?|)4b)_HK!"
    "-+h|N(LY7fM!k{1sCVlWhO1h<~j_Lxry+UUTBv7I=rt50VXjKxIQ=J6USLi?<>b1X}@(C$E4aj<90GdynhhY*-"
    "RhQJwWbF7gCPNGfH}kf9@pR=vf2Tg1B@5ptZ{580ItPUE+Kk%bcQ~^7aTHyRSnIC})r`12|*Pz|0E4_bS-"
    "J6c4q7<|^uHMA=dNC*LZ!dhf})_B0jL^h*^y+Tm%q@e^Xa^nz*}=Ka{|kIJc~1xW2HR!M0%0-"
    "iAEh+wwN!AtnS`iZ<d2XD~kFDKVC25JiYQaDpEGG_+oTSb(Nq}bjau1CtJ&66mAOyLym`@{e%MFLWvI-"
    "`0n1|U_(fI%Zh89$5So5p=m$(Gxc&Sqb;Fkvl~;C1bwS{9qlIj37V6x@4=0b7U^!#Z=2^DGaHGy?K{OoUO80)#sYwXKre3E!7+6S"
    "u*-F;0_}!F)6P2ZTa)A{E$ZDAD&ZBLKlHBk5~BN-"
    "~pE`CO6!3fyRdxxRa=l<VsxKz9HN+O*(IWE1do?%!B)<agE|aXf3gc>;#STUx6kq-"
    "5N~z9jXTI_y_<fHh10dNep(qhU!L!%V&0O+cK6xaAQ-2DdF6Mi|G-bq>^+5IiS1i!rW9Elr%t*xAh5-GS6P-"
    "@!<>G$zXmN#z>})wej=Wlm7mrZeZdRqOM}c*7>Ljf6#a8MUAW8F_wBbkdJ?7Lf%07$im<AlG(Hq-OPPWi1~h9HNFBEUbUV(_msfN"
    "(q4$<h)8JeWE@Gl32Q2#zcKk)|k_aP`s8`4{yQP9^HQ#xv}2-4ZT~=ePy8swWrSW-"
    "=!UEQv&TnaC3$qmQvnD6MzU(dksqSPg&)j9uQx1tv&fI$23gE=9o%Sc8(-ijbyY$+o-"
    "Kfj~bRBLXm1Z8lzepoVX3aw2rIYHpSI4rNz7&4A<-@<`;^}=X^47&TKfiac58K(@t^BG$<aDBT-"
    "V1$P~}B`|NIcN(Rx4=Q&cM)L49WT1q|I5aC71p)*^1pu;VyIx2&V<RMj|ISoi9YX4mR13iF|X|4PoPzXc9DEQ~2vk$dD*KhRfp;^"
    "$;%i5_RT}HX>knBaXJ@!C{e(6YD)w;JDi#KRzcuayre@g*v0)k@q?EDi|cc%U}mH{}Hu7cA5zF|$Rt=s)Wz>#c~Ub7SN0#mufKVK"
    "5oJ&E74jU5iJr|kP)dB|+_!KT(MCssMZL*{j*P%)A<`qQ;<@M=18_y3alGE+X|l@jW7;S;w#!wD(*sau#}rB9d>4`kx*QK`oj1=d"
    "|>_}nki@yYyz4bOCP^faKi4^yyYgEyGSflD$cXK}|MmA5DN5Aax>0%g3|*YbQ`U9-"
    "4r*5IC}?nY}K=A^rXOgG?ndk%2{&^@~IXduRJw?_xrMu;U&@lZ<7b`V>iNz$$;uM6IyCO_l+i&VAOQYtD}Raj0PQ}qnD#N#<b)xS"
    "h^Hd&G?zjhK<?4y)?o~w1RJ2UWhYPiDQ?u)gg4T+_+wBJ!{?BeN_L^Ji!hmv4b09fN1vFrVW=B4IeMiVsU;3H=|9-"
    "$Vc#(?R0vSM$ocfh&Csqm>%m}CSk7zcWGV5hMvB--in?E^NIri>@&TL^~;jLg{*F}00iQVR(p8y+dDn;xHqX=UX-mr>=xYf!pvQX"
    "a29ULOhRz8pM{C($SQ^Fi3}YgcpR|AgBc!E$wPQ!O}Lq}M;KZKsDv5DIbSqI59Keq{gYRGaE$Z#+2=MKEODmY5Po!n%ryIhUo5R{"
    "fCzHCa-f!|ryT?Jn0a^?D~Xv5l8>NJsy{*E-"
    "XO{5~w5oF$?kJtj36a<W*W<36uJP=VRg5}Uron}e9(aGpo!IX4CJUfU^s@cJ4@i>Sww7KiibgM(bmRHIw8zv~cl?m^>RxtjH}?<b"
    "vIEoYHZBdp1Dcb2dnf0w;X!pgvO9wVs;kQ7i}YHrDk!^pr2b#_~p!GJfPlkZ5Jsw;htO6Ls5K9lC&Z5+(xU0;F!bh&|MLKHRyLEP"
    "z({2%aKS8^)5ug`;<?!lT9Y+)oF?aU<!1J8cS5K`h_C!iZi4Qo?FEMA<>4IB&gHUfe5?A6%GUmzMw{M?0cuomQN+g5teKAZ1L@@8"
    "1S>L_(aoN;Ep<H78*agB{zxaHCGHC3!2^F*=iQ|Ez#MQq|frJ3$+?15-"
    "=OvNaFCg{dh<PBQuj7vDaD|EFVmJyI#BkVoLwt3WEk92R35V)DnmTlbsw0zqTKt7jvMqwx=r}5!>pfz#Ez&#nPg(=~#QQ+?E3Jo|"
    "pAx6=UMQja=54o6}fi1E0jdx-jXV^`~dvg)4WM}g;p52AbIhBldo72ke^6e471&tXTYas{xy=I^<yZSDi6SkvkM3d9nCgeZ=$GGa"
    "rX5AA>kDcY+J=>Ba%rlB?w(@6Oro83-Dgn<iJqy4&$zHO?)~EMh#WN*3-"
    "e|c;>{EDY&{680;YV8Hhfa5D9*JgDl*A5l1WRPN1elrUaBrD3LKjvz?W0Ago6;nmwEz01<CO0?F}JGRsQTG+MSWe_f7Zscn*StI$"
    "XA_RiaXG-8&lTJ<ItCob>$I~(TI`dE7l9U8g1SKQ_}f4PN-p?4Gh|-"
    "FcEY|!$)%*iQ$!vWGu~RU}6C)K9(`#z8t(o?2;9YR25{(xJ!$RP2w6*j=kj?Q(gX~Fl1wSE6uG%bFx6v8Wn&`w%4IGV2t%>J<(%}"
    ";Fj_7O6XIU5IOF1V{0w-"
    "vDQwMAf6=w(XDJZm%2aED66f#z<UJ9L9Eqn74nA@s#fnv{W9*!IdF@8y(OuM*`2{LrXJ{N=Azi#Yv@^fIBuK5`-"
    "!mXfdK&uChMZC>`({8e0GCr8bi#p#L4NQ*n+yxa;4ucG|>>cOx8e1qIx(fJ2@~TjWhmU<yVITNB~#^SsYzLQ`X$LAn;qaeoY+ri2"
    "MmO9wY3Sgv#^)>8sPZRD23_#mn&hQ<~F{07D#g1hCs?rQYNH?8ch;Wi*-"
    "V94~xVDNaq4b64mC<6z)0uS@f6W7*iY45x`Ot8J+=VKNLjd>J{e38wY6-l-"
    "@1ItGL@EATX!0~0wY*Dhp2<3r@ADTfXJt{!*pYgFcpI>)2VZ4*;ARcH{z^VV$uHqANl)MukjA4%v10oW5B{uuf??wY7%=^u{a2HR"
    "pEJu}U@_lky_&+)xD=mgTy&}l7*m{g>#^)*u>AGs4zJO}tQ@+yfvM~$e#(6p?L53XTtpw~=_+m=pz<!vZE?CY><*V@Es@NTrxU3l"
    "<05*6H#R(L+5GH}JR&4_(2-GQUfM?K+o(1AQg7-"
    "a5IT{T^*=T2f#QYkDwxhTU)nu^!n2Rs%yC5H=t?Jc_&uE_*x0J{u6o*_DBrYInXxx#Oq8bb^N<{n+m2u$U%j*`5y;S#sPx*BP6`|"
    "8m4wxr}~t__*Mm1g3c&1J-nGsKo+L%Gvk1GmExd-"
    "O86rc9bwz=x^Ojz*SY31ZvD=+^Vj``ddPl26r<L(Ukwd1R626M8I_E}6Kk3t@8v>RgNM(#@FkiAnwmVrtvRp0fD|%5{$TOpljFwB"
    "J+JTYFj}3Bj2>$Hx$7;&SlLE^D83uB_wNBe`~0V+}}I@oVJ3A3f`#26BLK!kC^yk3)n=NI83p`l$IFb~B;wS3??cR-Uo)yz9g(nN"
    "6a3wrk|2Sy;8z!TXNR;W<kG@0<2P?2+)P*sUcf^pVlABYtuXugB9wvT%=Y3216BP73A>EA*w#27`wkt*5eeKP(YBnhR#@cJj_TO?"
    "9{@H+ir4pefD55JAknt2yX|VY|Qy7_p>FT2;x5mP^&^o-z3zjfmb-c_`)WLbK=>c72!t%Oz*iJ=Df(?-"
    "Z5j{u<cZ?sGUOn<S^{SLE{;vn^ye(Ni;Tpw_Q&Fw!rZhxFZ8d0wKY8|+3?@`W4;p8o1ugBV9}%w|)au8v}SWajn=xVl3qS$QX_I|"
    "7lO;@}gAAdoeD^@>Yz&831PMFZ59Dn66htjVO}8lq5Iu80W|*w)*&A<Q(Fudg1j7)kN=>{r`JT8)uQCy5ZJs6c+~l~hg>p7u?|X("
    "~mEeWD+AQ3ZzJ;Ew~#eI|c_A5Dy4KXFMz%o2W@>nvT!ZI<8Y0jjGW5CW~vmU?v*^s`%gPk(E$CiI(Jn&70Lu!;q041nwi$hw<QIs"
    "5^dBbe8eoAO<7wUK<!uBQ<CwU-i|IGn@!J>YDH{M{Nz&_z%rYf_i%^yzx2zwZ;?xet_^Nw#&%5T&QCXzTX}TN3V$P9na-wb7MHa&"
    "o&P54X5RpvrH++GWl7iv$=|YdZcBu1;K!qzGgZW<9k=KL*uKOC#a%$MGb%r&+9$ROQ^~j!N(g&`-"
    "sXOo=fA1f^47JZTKBGZ_*(@oB2mX@;JeD-"
    "7c47z9RrOg+)e7)C9}D|RGB*Q46USAj!rgvC&*bclpO8856Qoq9H+qJU3;Nk@2C9`~S$&3u{;t@6cY3<mls`{zC05~A!d&?uY3mL"
    "IrCSjl6*Ftt}SUPWtlCmt%TBRSFCycLnRCrNaF`o|p>sxqCgK4%5`ZCbd6DUijTpO+5F^mg8wMr3|lY?<Qtj)}iGGGd6f(<Ew2D8"
    "~%a)iMyUqG%=pLu`sePELN=kn~%A$Wj*ppibCUJM1O``Tlj#J0d`61Q@0GmglvbDUi(cJ?9*#SX&NZKPiW5Xg8w4BlwiS$Wqbc8^"
    "|oLYn#7IVP2%8E#YPC4&BeZQ9Jn_Jis+Z=urd3-!TtCyPhSqzVsTciH-"
    "r7;{+i5@E3Kbn$`uw1jafIjs2V@^VAGo`kD2V>AvqzG@%haLK;P=m9zUXo}zS}q#sXh2TLOf^y$$_*k{0PNtWSnt55>)GBj|(J7b"
    "{DsR2o^T{p6x)sTea*bK*Wa=C>}2vEOf7w*q86z@T<n&phV`6Da@3*9*8d%iQmQ($x}<8pjW&Yp90TlewXEV@**gNc(=Mh>T9dG*"
    ";w$3}@I$MYCk%<6Aquc;)Tvsi6Zd`;P_O@-"
    "RyjY+Dmz{_EovTh!0D74FUjg*G{NP6^<0Q!~^;C7?%JtDpAe{Y$QD;p5wuvILH#ek|{yBLS77zZ~&;@41iFFD<)Mvvrf{m}B%Gq6"
    "cz(x7;zP~01(^ypsO9!v~<)S9sL{^^gvEiEZ?4TBTz$oH+QGW8y#aiyBvDtV`^+kW?i5uO3LcSc$WjJp0h)=g3hb9ql{z%;_x%)K"
    "nL&YBkKdQw8>qlQru51b(y=8+J;@F1JLN=>knsw^!us(5W|O%L_6Ns^%AwHm5h$9ZoWd7QY@FyjVS3)bT7K`#HyAtmaHVQbfQ>vp"
    "8#SFbp>yfK}MTNP<jGx+GP2x~XQh(^$(;3F0mTbFIKL(Vke%xU#W1Q;zty)uOyM}WHZwAEVxAbzDaGV+#|H0dM)&ExoTCDc_V6bw"
    "|K*)089f{!6l^#?AlEQMDqaUd}vnG?7)jnoIIZQNr5x?@btEjwpITS$~1El8O|0)+}bmgeNoNyfO+jy#X%rSTufvV&`IP6H!i({*"
    "{HPaN3yEo9V3P@ml_%Ze;49lERgnBrvB+@@gDIASLQ7zX*Qi1}ogQ|z#gEYI*7Vxj04)nkudB(j9fkL1Uzp+;Qv6&m^!1IIBUi>6"
    "p5Xyvw2&9d*9jW|~xZPMdjqjg}#!ounfh&yyoten33G{{7=T}GYEF|>C*J;yf~5J~9-"
    "2rgFaN;=PL9Bk76r0>9%*haGGE(6Wtuw4yEk2uGgqr&f7H#w?P><jt{ZNoCOT#{BQy|&~UMv>NKdCGg?CYLCMHQz9BN*C-"
    "Sz2FFUTSCNjmw09O(Xc99K|ph>C|7Ha#e+y}tCv4@LA+{nt7Uj?yG5AD9rv1ca5KK9$f!ejdPGH1VxhUHJo-"
    "(Y=u5(i*bjd(jZIyCaKt`S+P~dtog$b=PI*Ki(6~|lx`c$}q*tfs)8h3|tUk$lwBoSbVY$!U8;|aBiq_{oja|(*ka?MQ=O#L`N(x"
    "sYm+PqoHHtU<>dLm%u9V0*AWq4<yvBsZj2haEG<RwSpUNky2GQMPQa1y<n`WQG?-"
    "aSC++cTbZ>>~6*71Nud*+fyX^pUg5vGoTP9+tn82nF14KgmsLRdSpF$d-"
    "NBQK!4K7edtC_X|E^g!07nG#s)x@bKPc!{(@Sfg0%wl-"
    "y1)1sIZNz*cBshuH5JFKsr_@qPf9ZG+2cC=nN8=f{BR4*f5O%>|R?oB4^hUwTTYa5h6>WU+YG)lpGV3mH6u;>(Zt89upXh)d3rm@"
    "+u+3J->8Qfhl$}mD$v*C4QL7lQ{=ZNgXnr>Oa5xv%Pu3L;1DPM~aKsfn&MdWN-JkQR@d;6$$iUCNb5hGz<2&uC~*|d&CPFZn=bnH"
    "Zr0oRZoWDnT_tsj~c1KZ^Nl$DLVCr7(SzyPBd&#H~*&os4;oA(SSXT|<}ltUTK_bx)FOrRc>sw}lU1AL0gqtzbCk%Ufh5FYtRG^p"
    "sBBsbUDDP(i0{U$NYBa<h<I}=3ei+vnnjC9G<7m}7%Zl398v&Gs+s<2{&)@sRw&n{7e4|(RvJqnq>Q|y<3mH939b*yz_22bZ~(y>"
    "S1LUWcgA&FYX&Hm*nVb<P~<WlO(gzlb!kX$t?P~vHYYFa++-{GkR_6VL<<CROZ)Q;}y>D<q7k`+N{-"
    "Yg}x9p`Xrva?08GU5WAGqyM-d9)fRt%_mJ)OD!Sq_e{rvL;_hH@%@zof`ES=C<7JNk1nEU-"
    "x~4<~Yu5$xq+AGEwOWO;4W2GC1QVzGo79Wz^LnYYCZkjV*N=O9R%sIfR2&5xga_A>D+u+y13!mX9PVT+iT5s;qj#*15gyan2xSnw"
    "Hm1zf>BsUV}>O>|1otiBAvonucKS%!JU5dv$0Baw6oJ>X_T0+fbevtx*%;7A4J7vi#f-"
    "rgdGCYd_!j+<k(e^36t*_*zIM_Jr=!u3BgCUZSjz<{T3<UxHMp?o(*5pP3`T8ZDJW2$~JIX8XKv`95}UDUkFlwF0WeLeD{8>$V?m"
    "DpGd@_c^wbk$HJ<p>i>;qDtEmk7!9c$N>j*@cZPfVwB+CoIC9>^8pDTGL9;%#KB0nd(Q#*Y%+qqC{Z3JhfTZ}MnMdYNH^T#TkPI$"
    "97DlcV-"
    "o$GsBsHnTdM(+*zCZrD(DceQkAQzWpdcz4`mt9CRy6f5gF2wB)~Ofh2n(;;$&RqT6fnjE<SAUL<H<D?AjebLdPHTfz{@seE4pxKL"
    "t#{`F-#0&3#znJ0uCP<ds!d7i!TRQ5x>uveD%UZ8_nMYmYG_8C;Vm(Lsk-5=N1Jwc|#K8s-"
    "(39FWtHGpE=)*Upe$29pUSo$<sYyo}9pi;c|U*o%wv|COrQ!ZkJy!=5~C_YOe7R0}A_5HYyAuSdd^JqJ$By~^NE)nrx01??Qd_d0"
    "q4bb*v9`ooWrs}{R)z;XogBOQUj^4>YUrA_q-OEJ+s+MT1?bJ&QvX>Tz<%lDL#YemO5!cJe&ti`!{A44^IdM=ua4|*6*BfsLgscv"
    "YMShB$kvBGTGMuJh4rfPzD8u&5oI5SN5*V1mS8MEJbGO;{k8kO_>oK!(jt7@N;DnoL3{3;kzEtJf4vf8Jf|B7+g2}*xsocWFtt0#"
    "$aAA83Yv>J}Y@l1eC)g#CUs%WjHioi=-+)Ov)p%OKRQ0q~<uA!v}2!hpPjx3=IL6xxn7mEKde@Kd_X?1U`R|ghZyDnfNL9!iZw-"
    "*vI>ggI~Smk3-"
    "KNBN8vYU^f6Oha$0vCLe5736mFS`v56rvnb;*`X>dXaNtGNW|A#(ohBirZT2V;ZRikGW*Qfe)=C^)x~TA?nm&1MGp5GXv(D!fHX6"
    "=K<eI@Dn*yKWMC%dyM~kBVb5=gqV_U9~F`j6A9f`v0!@aA+z{Yz@9no_2FXW%MWnF3OE8OVwFs=&a^(Wt&*3dqG{guBQf-"
    "X?=f8mcUBKBW~RwJ^JrwM@w32S;i`QW&`nN%*#K-"
    "QcGkX>ePK^j^EjrnI3AhdoZ9On9TmPvZ=C&TO!pWo7KN_FtGQdFOd3f1hDMyCT~Gx=<VdeE)qU?|M=jIPRhFeaxiYjCVboO{MI88"
    "%>!YJloUSuYxhYh=4lwow_jC3LfZ-Y<37tVi503zsLP6r_adZf-r^|ANJ&HwXR);jS`A80zK@o<>-"
    "V)l#w8hoP^6KS91SzGXlgrs*9IK5@Yt{M71KXD^$x~<2oIBCk!aAA1Izq0^DOOtoM`!@)kWYoT&>dfs+NeF<SgmBZ%AB9BPOV;|h"
    "&uFU&u5<QgD&a*Sx3^xpF`1Hui2F+!5U|<3wajIDzma5dMHYGmVD9?=CLJwk;i<xIuYiP8h49`&Q39;DfSJbFs-"
    "u2@OiKu*hP(v?d(WNab-Fch<XLcLPXMd*v`8YqQIuf(Xqc8*N4k`f5T%=uwvsx<`DIW{#?hFw=EGas%^86_g1<v1!{rfahxL(l+X"
    "noj}Nbv^Hh%EL|9GhO;;FmBpl@BM**3q1ZNlH!uz=n0pXcYY>>2W#zZPpyf22pG4Ymgv};#{%{q`<8VLx%)&1QYmF*zHc%9_!y(g"
    "ZY92$M0i`B!_qM(|JiD0DECUj{@OXXTy05T_k++2$I<!lfZbjO-"
    "!acP&nvvMX$vk6s8m(e}V+G!OsTP?nTqbO6W<jeMGB%|5WB%ych1Pf^sb3)cLUTBLf3I+QG$L-"
    "K<o*8RCcyP=f>oGl||2z{wG^~t^{W*jE4nXS|!{kR@!rsPU%{zb>xg@Hi3q(|bAR(wWy4MQG#2l))oa^hXBP-"
    "}0LHBX=%Q4ec{5YnKbh;^VaDJ8b8m`3&c{iGS`*36Ko&2az9!VgYNBl9LxswtP^0&CW7=N-"
    "|H`GS7+ohuyko*wiZ{bBBk$PQ^HcP-(m`+J=ZnCb|o`^t4?-"
    "FfFqVb+tbd*?opU%}6N*!3$Z;f9u8IF08vKR9)Toso6<!ZV)Hypp=qxZ}_-"
    "llZn<@5AaJ>uccY1{!|WB^!43&gf{4TzjUtxHFBb@59iYKn0hC6@odwUF@Vq<;kqT?V!-"
    "RRbyAJd~!6aR3hrnENj3BUX%!bPGw^ZVvYu{<m$<%NVL45LkxfW$-"
    "lx*t^CSe)#}{eMy~la#}ndgPps^ewPC~vrW0Ql22<eqXo8B2x1~NC%D(lsagn;2QmETeMGc)eK8pxaYfWfUyW-"
    "(qNjRa^EQ6TyQkAU<$P9)C8DcXEf+H8o}prdLOdD)Y|FPgLA`B|uC!@WgB;1+cus{iI98~d^dwWcb1+$RIUVM(?$)czS2jfZ(jyV"
    ")1JoP*VGhCf({PU7YMO*(HkZ8ny<|;$P)k|bj%ruw<S7iD%OBm1NhA77kH%#>M-N1ppT6WZl9a>%JJTh*VGs--"
    "<yXUCnkqm;?Ew2|p-a8n!F@^?n_Wb8!wPM-NJfv*)<Ofv(VTbzex^Zp(2J(U<8?1Z?#ZXiJAn~6hTf2Mcn*7+R<n}Z&hA&TBo3}R"
    "`mx`3Po(B|norhWD$uJfV#op9IcZ8hhDM*gU-"
    "86Xdv@9DkqNmV$IhPZ@{v0qrfu1JB%{Oq%38fNgqq^yzS}cK+d5rMRhCmaVZ_x%Z&bxBjltZFhF^}^JT^K#?S<b0bIyD$VdLEkvN"
    "n2yIhclonU&>`72vjB$9?Od0~Np#Jd~YVKuk&D9DR``BDL7YxU8#Drp|Fev~7lo34LVC^UJGT0y_h}T&VdQf8xH6c-"
    "U6Zth1#jq@rdr)L3C>jkvZmzYbewziHU1;3pd$MI9Db)0B8&zFW}h<f#1DdF|<o4#*nOJ&XF2BPv+!OdQ}^^lg(e={N{E+mOxDz3"
    "B}ov>Z=c%dc=tE&++im}r|*yS{m~PhdM2qFy!O$)gkF7p5LHt~9mLLc4NXYP?%e>(=m%G*I<2_o0RI20r3tbY2(@&#Qp&N}8p+<x"
    "~QE6V$TvT5~Y$q%A5X*6^hsV)<gG2r(8t_1%&JS3NUz$I{6XV%R9g%?n%UAr%O{a!+*e-"
    "k0$@lIThnw+_3uF5P;D)Ugjo;F`UC))H1a4fV)(gid_1(W5l5RhmM>HUW)7Q?t-"
    "mGLrajLGSv9ngkV2S1z$a+UEuzX1LYu4{Dp8vkkvIj?Bv^x><G3=!W>2m;k@U+b=d4qfK<bqq}JmFsM(A*}B6JQd;m^^P@8wR-MN"
    "O`g7>Mb+Db{0Gh*O-"
    "JUY!ElRggZfm`FnUkP;g*8>S6Em+2kmsf?KGGdRXVB7|nkQTalME&MD%k(SACqp>iBn5jtwxpCvt{L|b>G?IuPWntND!|%PEUfLd"
    "WBbz<$kV#WK#?E2!S)4rY+!uh+)A|u}N>NLE~LdF&M->-"
    "liNS#DzY%M&!_$6v1Rh3<8JM9Rob9902#oAD$77dq!Jo<^j8tAd|r*3rEYDbX(+fQ8!Ml`#)opW37X`s0A#1cY1UE<E2gfiWPG1G"
    "rf{-"
    "4QR?*yT!T~K@arrl`i<ixH8l(YnFGeNB^X6JVC~HQJbiiX2+PIgOjn%ZgcM>d)0EL<oj5b^kTL|Can77&$xSJFf%SN>_>&T#@7mV"
    "q&xbYqy=uZF9~-)m*H-"
    "7I|od?NrJ1@DnK@rKuLVtu>FN5BpRCtsJ2FrfVyUOKG|B0#NCk$RH+J2U)hp<Ouy^Ci<j2w9!Uvu9v=={5?#;2R-"
    "=TLi%Gk<mjLhnyp3#3v>^6iBggV_O|g-"
    "u5t>fp{M3b6YjK;V_Gm{T;m$PPxmt5N#V<kPm~75=aY2a^t#V7vBTPB*VYYfeh(7ZuIm@bPRN6bWVV$hAEl|^ECJdSKk?6*DXj_y"
    "_O~hcTRF^;h`G+?@z4@`o=HVGmIwO13thE(SNhna7YKT_}vL`zD5jDlq^Qo#6<~1SLUFF_4Mfpn&xQYov1e%b>E#s{|pHT&!cB4m"
    "@hS6-"
    "<rX1(uWGH==E)gP9h=NP2b*5lL!UKD5aGV&aqMbx4yO*qrdKx2hZsn5mQ6OI#%(T(U1b9hS8gEIj2dmiSGdQ=S#qX28y(pgfQ5nG"
    "<45L}jOR`SoVQ_pnLg_YjT%Rt+M?_eA%mHhY+E1x(XVNicnRR99_C3s;?q#_XQSm7GV`Iu3wy{Uj{pPeG>oD%JVwNwE0?Wv((4ys"
    "k*%MtkN@+T@zz3m7YuqizcZC9)s(sr~nO(#sd2o>BHS&ZxMVCgX?TW;aAc+d3qDv105`3H)9D3V2rY$_{sEwyp?fCOQfB26#|G4i"
    "=ZFE4{gMfvIDhnFt*L~%Z7Z$H+8Z)?GiayY!da!Jc$tp=OjZb)3&X5#scF~_v2^|G&!N19lWR9SYdj$C~1raBjF>InKgVOsmqFup"
    "l{Le8}D7PiK79nGty4D6!no8mtY&JQ+?gA`mNWXRGm0OO9rHJnOGa0@Fv|Eib>N^6ys&CG335Z7w-}if;0FC;FTo-"
    "#p6pscGF3GL9d@XV8_!;*tf*9jSb5yCcB=7H$+|i?nxR{YkxEzL6r8On$VCkcJF8P*!8%u~o7lB)@1wX=^(8Vi}06hi53DdL-"
    "&NTNiFh3;_M9tjiBV%GSF_a;=YyXx<d<}dlX<Ac$-"
    "wiG=o|4s2CHqG*1GQGpt%Hmi%QEM&^gLQo6||K0=i&r6Ph)YcWmhjyVdg+G{rR8w-"
    "~ICX^7ZeZU*G=v`tmVf=C`bx|EmAcK1wp1|JzSepZgC1zFWVJeff`QL^FsybDC}eVN;XLwaD$Md5Jj7iBzBSv^~yf^7{<1C;h~(O"
    ")63wJzrAHhK`F<sc|vW;^wo(&yW6Ggr1Z-"
    "kRA0ooE7u<42RZ!d8h(zwbYCCImZXSGD~`?i*|gD?C!@jkb!aN4s!B&^~>YT*K;XWQ+*E3Nw1oLzOExJ-"
    "Wj!Uth@Cy7CWZYS7foWb}H5nSsEu*XH~4UN{`e4B!)|y*A%p($C^Hup0>;gp7d2>dyRKP<qv9_b-"
    "dVAG*UUW(X!5s#IX?xNQs6q`-m2^ZH2!&Ym(DLgp-fp+y?;A@R*a4w%m=?aNsl17dd!vQOBTe>(#5PGXd-kQeFffM;17pJ#Fyy`E"
    ">-i#*M@}bh|w(q8xSTp9|ol>q*L-9HVyB1=hQ()J0v^<F3=}-MR08wF-"
    "h5W!8weO@Q(PM0&GRo6!@m4oZt>)3(f$Qd$=#`)Fg0CHH=;+)T$6a*>&>JzYZ}_{!ijn{PXZ(ugbR)K;zQ1adB0w4c{C{81>Y#c+"
    "O9XDr0Bq=l+rAI%a5r9q_E^f^XI(_X)r_FJ(FRRK$&)UmXeo=LBKo+5#{>hb2k@-"
    "%|t&f~i!8dOK=)Dj;ZD~If0HT?TJDf7@C8kUzxLB9tQAWFxcZ1;BTE9HQi6S~nEX%7x6<>ZnK+3px5>Vm;=F6n~oT&*}7$73W?C7"
    "MJ>W8vs+9;lX`_8Re4ZKRR07HJp@BriF`fHW5ECQ)i-"
    "Hniqb+g^#=zKiLd8m%Fp2qQth?om@7&3`*>6p|6<wtkqjiKm@2U8=2>{!#X*>bZq-TQF3NLsY;KU;N>k`kxoFT<ai%AEAZzoB0P9"
    "=Z~)9A6>8tPKtrfxF&CL`r7xcW1l`tYi7!$u-"
    "ghGw0^hYeEIq=`;D$hP$QI`WvKguPs!mhqF0h^=kQ7X*U!x+U13_^N4R@?I8#480+N7LU5hfsMJGB)WACi1hml4a<CL|RK7$)U(~"
    "vIf%6oRu?_pE9ShdPk`WU1H&p=xJ*1SiUeD>NDSvlk>A?I+^k9_K2TFa!kyO6oiaSiJc=P?lY0M=%-"
    "dmdRt*|qU_^#ZAr>L<>7ykbyhk9v{`_m-tXu(HBs*k3J81$=##5#?%pA_^@(@!pAYya7Bet}wibUU+48PC2raSxn5ZGjVQ|SjI>J"
    ")Tdj9E0xiL=O0#f{-qO*CL^?FlfdqBJIx9n<F!;pPy%r!3Ll&LnTRU)O0nZIr&6_GO?uz@-"
    "Fv}t`NaxPVWBMoCF;6rX|E`8wQ<gVgpalxFM|V68;Yj*C`X$9v%?UJO<M^0GOjhYK037{pAs{?B&fKZ^Yc;&eLQrS-"
    ";|jr6U`O7+fuE_^^6H6K~7`(7?Xas)1o-"
    "7RRQjg^a7{iMb%nDY^YXBlz7>mv{fm)&mi3X_<0I&mpn@XMQ*e?+B4lk=Fg+J9y)^e0d6~Ei>%Kb3E6bg0vGKdVR#s!VoCfe7PT{"
    "lZQsP&c&xY_mkGXe@3w6XjutqQ1Y0BNTn7y>3nx~|`urulICUL%NCk>TfIK4{`NggpscETZ5I6-gvDN3?Tr-enZ1ENasZi#VyJ8`"
    "4G29pJ#OYVL8iSc=-UJbubx)%ZGq^H^+sj&$F(~~SUHnpH^Ci2g)d)Olt(LoHgW(6r<lY=QVSMY9WD~0S-"
    "2K^I*<jt*;jBT!xG|t7aDMjOoF}h>VOAF2yeHdQOyQslraIy61ag!I>W+GpNP|r@=MXx(DFROwYu5Z2rDFs+#yM<f*}~=ooV&mGL"
    "|;z-&5HV8S2G+i)yo#CRWC08_2T@|HT<Is21pBXL0xN7-"
    "@bivs=(Wi(z_dEY%%Q4k^CO_HuUxN*WX@0XDfR1<Mc0&AKrX<`Q`rO*Ux`{{o&28uU}u@{P-"
    "VlUOxQg)93eJ|N85jAK(1;+ow18SG~U8`<7o+Mg7B@ckjP^{c>+)Z{EHB{`R+*&o95;U*yj}fBN;e53m2HKl=3P-"
    "5Wpk^UM2>Z=e6>{@#Ci|MBI+{Y4*MzP{f7?=PR<|GGTxKm6nUJ0I6S_<eo;@wb<+Utd3eeEZYq*O$M&e)dg2-"
    "LLH9>&LHe|MB|%FMrK<{{FMSefsp_4?g8P{q@a{AOHU0!}ovldcT~n|NiF3hqvVU>0ds-{QUL(r;q(s{qp|x!@Do{7k|$0a=#z-"
    "Mtpk(|MBg2tACQ;um0G-"
    "V||d{^T*jApTFz(AO7dx_s8{L9{>E|jaPEtJ#T)0`SA9SUX9P+U*jPE=CAMHy?b@Iz3*ppl{X*mZ?nDr>3;a${QTklZ+=<-"
    "<d^%Jz5e2(`!D>~<HP%Ie)H4MZ~yk;+h^{t`27CM-`;-u>EB-Ocjou|cX;#hey`s2=bvBLAOG*azkK}q{=;kktaqP3z0EC_pXH<P"
    "zo0+$(}&lO@4ox^mk*ylefII>-"
    "TwUk=f8e<^&&t1e81THR?hXwkNoBH>+Ao0y)W~J_doyp+xwRO__sGd{_^tS%c~ps!~Kfy^ZwWSJKV4Or<V^e_jh>b-"
    "hTQ0_3g`t``7&I>(77t?bG{@F6EoguV4QD;p>-"
    "w{rus+IB)CLpU;2&?I$1JUw41eeNkV2{q^N}t$+1GF7NOC((dc>`Sl+!pWl6XzKH(x%e&w2-}>b*uV2Q4cVD-"
    "^zrOwH{mcEC%`g7_Q~Cbm`|obC|L%3&-}*7V{q6H7H^j@&|MG8N?)T+e-TC|L=P!Au-n{?q-Jd?b{{8dIhd=dSe)-"
    "ey)BU9%Uw(Bx|JVQc)BmiyZ#;dk@8167y{9IN!9V}`{{U9<iEa"
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

class RLPattern(Player):
    """RL-Pattern as a standalone Axelrod strategy.  The 85-feature MLP weights are embedded; only sample mode is supported."""

    name = "RL-Pattern"
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
        _require_sample(mode, "RL-Pattern")
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
          name: str | None = None, **_: object) -> RLPattern:
    return RLPattern(seed=seed, mode=mode, name=name)


__all__ = ["RLPattern", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((RLPattern(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{RLPattern.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")
