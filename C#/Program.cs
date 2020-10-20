using System;

namespace Brainy
{
    public enum Ops
    {
        TRESHOLD_FUNC = 0,
        TRESHOLD_FUNC_DERIV = 1,
        SIGMOID = 2,
        SIGMOID_DERIV = 3,
        RELU = 4,
        RELU_DERIV = 5,
        TAN = 6,
        TAN_DERIV = 7,
        INIT_W_MY = 8,
        INIT_W_RANDOM = 9,
        LEAKY_RELU = 10,
        LEAKY_RELU_DERIV = 11,
        INIT_W_CONST = 12,
        INIT_RANDN = 13,
        SOFTMAX = 14,
        SOFTMAX_DERIV = 15,
        PIECE_WISE_LINEAR = 16,
        PIECE_WISE_LINEAR_DERIV = 17,
        MODIF_MSE = 18

    }
    
    public class Dense
    {
        public int in_;
        public int out_;

        public double[,] matrix;
        public double[] cost_signals;

        public int act_func;
        public double[] hidden;
        public double[] errors;
        public bool with_bias;
        public Dense()
        {  // конструктор
            in_ = 0;  // количество входов слоя
            out_ = 0;  // количество выходов слоя
            matrix = new double[10, 10];  // матрица весов
            cost_signals = new double[10];  // вектор взвешенного состояния нейронов
            act_func = (int)Ops.RELU;
            hidden = new double[10];  // вектор после функции активации
            errors = new double[10];  // вектор ошибок слоя
            with_bias = false;
        }
    }
    public class NetCon
    {

        public Dense[] net;
        private int sp_d;
        private int nl_count;
        private bool ready = false;

        public NetCon()
        {
            net = new Dense[2];  // Двойной перпецетрон
            for (int dense = 0; dense < 2; dense++)
            {
                net[dense] = new Dense();
            }
            sp_d = -1;  // алокатор для слоев
            nl_count = 0;
        }
        public void cr_lay(int in_ = 0, int out_ = 0, int act_func = 0, bool with_bias = false, int init_w = (int)Ops.INIT_W_RANDOM)
        {
            Dense layer;

            sp_d += 1;
            layer = net[sp_d];
            layer.in_ = in_;
            layer.out_ = out_;
            layer.act_func = act_func;

            if (with_bias)
                layer.with_bias = true;
            else
                layer.with_bias = false;

            if (with_bias)
                in_ += 1;

            for (int row = 0; row < out_; row++)
            {
                for (int elem = 0; elem < in_; elem++)
                {


                    layer.matrix[row, elem] = operations(
                    init_w, 0);
                }
            }
            nl_count += 1;

        }
        public void make_hidden(int layer_ind, double[] inputs)
        {
            Dense layer;
            double tmp_v;
            double val;
            layer = net[layer_ind];

            for (int row = 0; row < layer.out_; row++)
            {
                tmp_v = 0;
                for (int elem = 0; elem < layer.in_; elem++)
                {
                    if (layer.with_bias)
                    {
                        if (elem == 0)
                            tmp_v += layer.matrix[row, elem] * 1;
                        else
                            tmp_v += layer.matrix[row, elem] * inputs[elem];
                    }
                    else
                        tmp_v += layer.matrix[row, elem] * inputs[elem];
                }

                layer.cost_signals[row] = tmp_v; ;
                val = operations(layer.act_func, tmp_v);
                layer.hidden[row] = val;
            }
        }

        public double[] get_hidden(Dense objLay)
        {
            return objLay.hidden;
        }

        public double[] feed_forwarding(double[] inputs)
        {

            int j;
            Dense last_layer = null;
            make_hidden(0, inputs);
            j = nl_count;

            for (int i = 1; i < j; i++)
            {
                inputs = get_hidden(net[i - 1]);
                make_hidden(i, inputs);
            }
            last_layer = net[j - 1];

            return get_hidden(last_layer);
        }


        public void calc_out_error(double[] targets)
        {
            Dense layer;
            double tmp_v;
            int out_;
            layer = net[nl_count - 1];
            out_ = layer.out_;

            for (int row = 0; row < out_; row++)
            {
                tmp_v = (layer.hidden[row] - targets[row]) * operations(
                    layer.act_func + 1, layer.hidden[row]);
                layer.errors[row] = tmp_v;
            }
        }

        public void calc_hid_error(int layer_ind)
        {
            Dense layer;
            Dense layer_next;
            double summ;
            int in_;
            int out_;
            layer = net[layer_ind];
            layer_next = net[layer_ind + 1];
            in_ = layer.in_;
            out_ = layer.out_;

            for (int elem = 0; elem < in_; elem++)
            {
                summ = 0;
                for (int row = 0; row < out_; row++)
                {
                    summ += layer_next.matrix[row, elem] * layer_next.errors[row];
                }
                layer.errors[elem] = summ * operations(
                    layer.act_func + 1, layer.hidden[elem]);
            }
        }

        public void upd_matrix(int layer_ind, double[] errors, double[] inputs, double lr)
        {
            Dense layer;
            double error;
            int in_;
            int out_;
            layer = net[layer_ind];
            in_ = layer.in_;
            out_ = layer.out_;

            // for row in range(layer.out):;
            for (int row = 0; row < out_; row++)
            {
                error = errors[row];
                // for elem in range(layer.in_):;
                for (int elem = 0; elem < in_; elem++)
                {

                    if (layer.with_bias)
                    {
                        if (elem == 0)
                            layer.matrix[row, elem] -= lr * error * 1;
                        else
                            layer.matrix[row, elem] -= lr * error * inputs[elem];
                    }
                    else
                        layer.matrix[row, elem] -= lr * error * inputs[elem];

                }
            }
        }

        public double[] calc_diff(double[] out_nn, double[] teacher_answ)
        {
            double[] diff;
            diff = new double[out_nn.Length];
            int len;
            len = teacher_answ.Length;

            for (int row = 0; row < len; row++)
                diff[row] = out_nn[row] - teacher_answ[row];
            return diff;
        }

        public double get_err(double[] diff)
        {
            double sum;
            sum = 0;
            int len;
            len = diff.Length;

            for (int row = 0; row < len; row++)
                sum += diff[row] * diff[row];
            return sum;
        }
        public double operations(int op, double x)
        {
            double alpha_leaky_relu = 1.7159;
            int alpha_sigmoid = 2;
            double alpha_tan = 1.7159;
            double beta_tan = 2 / 3;
            double y = 0;

            switch (op)
            {
                case (int)Ops.RELU:
                    {
                        if (x <= 0)
                            return 0;
                        else
                            return x;

                    }


                case (int)Ops.RELU_DERIV:
                    {
                        if (x <= 0)
                            return 0;
                        else
                            return 1;
                    }
                case (int)Ops.TRESHOLD_FUNC:
                    {
                        if (x > 0)
                            return 1;
                        else
                            return 0;
                    }
                case (int)Ops.TRESHOLD_FUNC_DERIV:
                    {
                        return 1;
                    }
                case (int)Ops.LEAKY_RELU:
                    {
                        if (x <= 0)
                            return alpha_leaky_relu;
                        else
                            return 1;
                    }
                case (int)Ops.LEAKY_RELU_DERIV:
                    {
                        if (x <= 0)
                            return alpha_leaky_relu;
                        else
                            return 1;
                    }
                case (int)Ops.SIGMOID:
                    {
                        y = 1 / (1 + Math.Exp(-alpha_sigmoid * x));
                        return y;
                    }

                case (int)Ops.SIGMOID_DERIV:
                    {
                        y = 1 / (1 + Math.Exp(-alpha_sigmoid * x));
                        return alpha_sigmoid * y * (1 - y);
                    }
                case (int)Ops.INIT_W_MY:
                    {
                        if (ready)
                        {
                            ready = false;
                            return -0.567141530112327;
                        }
                        ready = true;
                        return 0.567141530112327;
                    }
                case (int)Ops.INIT_W_RANDOM:
                    {
                        Random r = new Random();
                        return r.NextDouble();
                    }
                case (int)Ops.TAN:
                    {
                        y = alpha_tan * Math.Tanh(beta_tan * x);
                        return y;
                    }
                case (int)Ops.TAN_DERIV:
                    {
                        y = alpha_tan * Math.Tanh(beta_tan * x);
                        return beta_tan / alpha_tan * (alpha_tan * alpha_tan - y * y);
                    }
                case (int)Ops.PIECE_WISE_LINEAR:
                    {
                        if (x >= 0.5)
                            return 1;
                        else if (x < 0.5 && x > -0.5)
                            return x;
                        else if (x <= -0.5)
                            return 0;
                    }
                    break;
                case (int)Ops.PIECE_WISE_LINEAR_DERIV:
                    {
                        if (x < 0.5 && x > -0.5)
                            return 1;
                        else
                            return 1;
                    }
                case (int)Ops.INIT_W_CONST:
                    return 0.567141530112327;
                    ;  // case Ops.INIT_RANDN:
                    ;  //     return np.random.randn()
                default:
                    {
                        Console.WriteLine("operations unrecognized op");
                        return -1.0;
                    }
            }
            return 0;
        }

    }

    // class Program
    // {
    //     static void Main(string[] args)
    //     {
    //         int epochs;
    //         double l_r;

    //         NetCon net_con;
    //         double[] inputs;
    //         double[] output;
    //         double gl_e;
    //         double[] e;
    //         int ep;
    //         double[,] train_inp;
    //         double[,] train_out;
    //         double[] output_nc;

    //         train_inp = new double[4, 2] { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };
    //         train_out = new double[4, 1] { { 1 }, { 0 }, { 0 }, { 0 } };

    //         inputs = new double[2];
    //         output = new double[1];

    //         epochs = 1000;
    //         ep = 0;
    //         l_r = 0.1;

    //         int single_array_ind;

    //         // errors_y = [];
    //         // epochs_x = [];

    //         // Создаем слои
    //         net_con = new NetCon();
    //         net_con.cr_lay(2, 1, (int)Ops.SIGMOID, true, (int)Ops.INIT_W_MY);
    //         // n = cr_lay( 3, 4, TRESHOLD_FUNC, True, INIT_W_MY)

    //         while (ep < epochs)
    //         {
    //             gl_e = 0;

    //             for (single_array_ind = 0; single_array_ind < 4; single_array_ind++)
    //             {

    //                 for (int elem_array = 0; elem_array < 2; elem_array++)
    //                 {
    //                     inputs[elem_array] = train_inp[single_array_ind, elem_array];

    //                 }
    //                 for (int elem_array = 0; elem_array < 1; elem_array++)
    //                 {
    //                     output[elem_array] = train_out[single_array_ind, elem_array];
    //                 }

    //                 output_nc = net_con.feed_forwarding(inputs);

    //                 e = net_con.calc_diff(output_nc, output);

    //                 gl_e += net_con.get_err(e);

    //                 net_con.calc_out_error(output);

    //                 // Обновление весов;
    //                 net_con.upd_matrix(0, net_con.net[0].errors, inputs,
    //                            l_r);


    //             }

    //             // upd_matrix( 0, net[0].errors, inputs, l_r)
    //             ep += 1;


    //             gl_e /= 2;
    //             Console.WriteLine("error {0}", gl_e);
    //             Console.WriteLine("ep {0}", ep);
    //             Console.WriteLine();

    //             // errors_y.append(gl_e); ;
    //             // epochs_x.append(ep);

    //             if (gl_e == 0)
    //                 break;
    //         }

            // plot_gr("gr.png", errors_y, epochs_x);

            //     for single_array_ind in range(len(train_inp)):;
            //         inputs = train_inp[single_array_ind];

            //         output_2_layer = feed_forwarding( inputs);

            //     equal_flag = 0;
            //     for row in range(net[0].out):;
            //         elem_net = output_2_layer[row];
            //         elem_train_out = train_out[single_array_ind][row];
            //     if elem_net > 0.5:;
            //     elem_net = 1;
            //     else:
            //         elem_net = 0;
            //     Console.WriteLine("elem:", elem_net);
            //     Console.WriteLine("elem tr out:", elem_train_out); ;
            //     if elem_net == elem_train_out:;
            //     equal_flag = 1;
            //     else:;
            //     equal_flag = 0;
            //     break;
            //     if equal_flag == 1:;
            //     Console.WriteLine("-vecs are equal-");
            // else:;
            //     Console.WriteLine("-vecs are not equal-");

            //     Console.WriteLine("========");

        
    }


