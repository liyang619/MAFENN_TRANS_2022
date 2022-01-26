import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


# axis_short = [0, 2, 4, 6, 8, 10, 12, 14]
# axis = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# axis = [0, 2, 4, 6, 8, 10, 12, 14, 16]
# axis = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
# axis = [20, 22, 24, 26, 28, 30]
axis = [24, 26, 28]
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 1.0
# plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.family'] = ['stixgeneral']  # 用了这个才能正常显示对数图？需要再好好看看
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex'] = False

# final
rls = [1.180307249223876837e-01, 1.838228594667028204e-01, 1.950385894891698280e-01, 2.567728831337376394e-01, 2.348103192716331222e-01, 3.883304851408799774e-01, 4.801296909063029750e-01, 5.914514225776845713e-01, 7.232601396021716145e-01, 8.023791481200818199e-01, 8.516176918307618582e-01, 9.603527165978137381e-01, 9.911820850546564365e-01, 9.984177531650492154e-01, 9.980644143353341047e-01, 9.999133319851641888e-01]

# pre_crnn
pre_crnn = [2.021763016209392749e-01, 2.399543406873571039e-01, 2.908925376666172125e-01, 3.564600095748134412e-01, 4.417804816122739564e-01, 5.500235580832013582e-01, 6.704511004509089434e-01, 7.884316758744596143e-01, 8.715528671597795318e-01, 9.501043903088064457e-01, 9.811520597427938473e-01, 9.954151738477067246e-01, 9.972996099758273392e-01, 9.991071507519426742e-01, 9.999568943919971975e-01, 9.999569470239970670e-01]

# crnn_fb
crnn_fb = [2.025660256602566101e-01, 2.401962019620196287e-01, 2.912773127731277389e-01, 3.572291722917229073e-01, 4.436084360843608465e-01, 5.517651176511765287e-01, 6.707661076610766937e-01, 7.948155481554814994e-01, 8.977365773657737247e-01, 9.616446164461643908e-01, 9.880446804468044908e-01, 9.976093760937608801e-01, 9.994463944639445652e-01, 9.997759977599776082e-01, 9.999663996639966967e-01, 9.999755997559975906e-01]

# crnn_nnfb
# 1（8，3，10）：最佳[没数据]；
# 2（3，1，10）：最佳
crnn_nnfb_1 = [9.900953009530095716e-01, 9.984179841798418442e-01, 9.998249982499824995e-01, 9.999559995599955897e-01, 9.999939999399993784e-01, 9.999959999599996596e-01]
crnn_nnfb_3 = [2.024280242802428131e-01, 2.407624076240762467e-01, 2.916679166791668054e-01, 3.587525875258752439e-01, 4.453864538645386228e-01, 5.535325353253532921e-01, 6.758357583575835825e-01, 7.976649766497665439e-01, 8.972219722197222191e-01, 9.587155871558715736e-01, 9.878458784587845765e-01, 9.979029790297903046e-01, 9.995763957639576258e-01, 9.998717987179871480e-01, 9.999873998739987613e-01, 9.999927999279993873e-01]
crnn_nnfb_4 = [9.994093940939409171e-01, 9.999355993559936318e-01, 9.999795997959978200e-01]
# plt.semilogy(axis, (np.ones_like(crnn_nnfb_4[2:5])-crnn_nnfb_4[2:5]),   "C4", label='CRNN_nnFB(8,3,10)')  # 最佳
plt.semilogy(axis, (np.ones_like(crnn_nnfb_1[2:5])-crnn_nnfb_1[2:5]),   "C1", label='CRNN_nnFB(8,3,10)')  # 最佳
# plt.semilogy(axis, (np.ones_like(crnn_nnfb_3[10:])-crnn_nnfb_3[10:]),   "C3", label='CRNN_nnFB(3,1,10)')
plt.semilogy(axis, (np.ones_like(crnn_nnfb_3[12:15])-crnn_nnfb_3[12:15]),   "C3", label='CRNN_nnFB(3,1,10)')
plt.semilogy(axis, (np.ones_like(crnn_nnfb_4)-crnn_nnfb_4),   "C4", label='CRNN_nnFB(3,9,10)')

# # mlp经过了弱化0
plt.semilogy(axis, (np.ones_like(rls[12:15])-rls[12:15]),   "C0", label='RLS')
#
# plt.semilogy(axis, (np.ones_like(pre_crnn[10:])-pre_crnn[10:]), "C0", label='CRNN')
# plt.semilogy(axis, (np.ones_like(crnn_fb[10:])-crnn_fb[10:]),   "C0", label='CRNN-FB')
plt.semilogy(axis, (np.ones_like(pre_crnn[12:15])-pre_crnn[12:15]), "C0", label='CRNN')
plt.semilogy(axis, (np.ones_like(crnn_fb[12:15])-crnn_fb[12:15]),   "C0", label='CRNN-FB')


plt.grid()
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.legend()
# plt.savefig('com.png', dpi=1000)
plt.savefig('com_crnn_nnfb.png', dpi=1000)
plt.show()
