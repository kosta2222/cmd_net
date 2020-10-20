# nn_constants.[py]
# Параметры статических массивов,количества слоев,количества эпох
max_in_nn_1000 = 10000
max_trainSet_rows = 4
max_validSet_rows = 10
max_rows_orOut_10 = 10
max_am_layer = 7
max_am_epoch = 25
max_am_objMse = max_am_epoch
max_stack_matrEl = 256
max_stack_otherOp_10 = 4
bc_bufLen = 256
elems_of_img = 10000
max_spec_elems_10000 = 10000
max_spec_elems_1000 = 1000
# команды для operations
TRESHOLD_FUNC = 0
TRESHOLD_FUNC_DERIV = 1
SIGMOID = 2
SIGMOID_DERIV = 3
RELU = 4
RELU_DERIV = 5
TAN = 6
TAN_DERIV = 7
INIT_W_MY = 8
INIT_W_RANDOM = 9
LEAKY_RELU = 10
LEAKY_RELU_DERIV = 11
INIT_W_CONST = 12
INIT_RANDN = 13
SOFTMAX = 14
SOFTMAX_DERIV = 15
MODIF_MSE = 16
# байт-коды для сериализации/десериализации-загрузка входов/выходов,загрузка элементов матрицы,сворачивание то есть создания ядра, есть ли биасы,остановка ВМ
push_i = 1
push_fl = 2
make_kernel = 3
with_bias = 4
determe_act_func = 5
determe_alpha_leaky_relu = 6
determe_alpha_sigmoid = 7
determe_alpha_and_beta_tan = 8
determe_in_out = 9
stop = 10
