// AutoSafe-MAC: 12-16-4 INT8 neural-network accelerator (single MAC, sequential FSM)
// BUG parameter injects a deliberate fault for the verification experiment (0 = correct design).
//   1 hidden activation clamps at 63 instead of 127   2 no ReLU   3 layer-1 bias ignored
//   4 shift off by one   5 layer-2 bias ignored   6 layer-1 accumulator clamped at +/-14500
module autosafe_mac #(parameter BUG = 0) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_we,       // write one input feature
    input  wire [3:0]        in_addr,     // 0..11
    input  wire signed [7:0] in_data,
    input  wire              start,
    output reg               busy,
    output reg               done,
    output reg  [2:0]        action       // 0 GO, 1 STOP, 2 TURN_LEFT, 3 TURN_RIGHT
);
    reg signed [7:0]  w1 [0:191];         // 16 neurons x 12 inputs
    reg signed [31:0] b1 [0:15];
    reg signed [7:0]  w2 [0:63];          // 4 neurons x 16 inputs
    reg signed [31:0] b2 [0:3];
    reg        [7:0]  shm [0:0];
    reg signed [7:0]  x [0:11];
    reg signed [7:0]  h [0:15];

    initial begin
        $readmemh("w1.hex", w1);
        $readmemh("w2.hex", w2);
        $readmemh("b1.hex", b1);
        $readmemh("b2.hex", b2);
        $readmemh("sh1.hex", shm);
    end

    localparam IDLE = 3'd0, L1 = 3'd1, L1F = 3'd2, L2 = 3'd3, L2F = 3'd4, DN = 3'd5;
    reg [2:0] st;
    reg [4:0] n, i;
    reg signed [31:0] acc, best, v, nb, a16;
    reg [2:0] besti, ni;
    reg [7:0] sh;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; busy <= 0; done <= 0; action <= 0;
            n <= 0; i <= 0; acc <= 0; best <= 0; besti <= 0;
        end else begin
            if (in_we) x[in_addr] <= in_data;
            case (st)
                IDLE: if (start) begin
                    st <= L1; busy <= 1; done <= 0; n <= 0; i <= 0; acc <= 0;
                end
                L1: begin                                  // acc += w1[n][i] * x[i]
                    acc <= acc + w1[n*12 + i] * x[i];
                    if (i == 11) st <= L1F; else i <= i + 1;
                end
                L1F: begin                                 // bias, shift, ReLU, saturate
                    a16 = acc;
                    if (BUG == 6) begin
                        if (a16 > 14500) a16 = 14500;
                        else if (a16 < -14500) a16 = -14500;
                    end
                    v = (BUG == 3) ? a16 : (a16 + b1[n]);
                    sh = (BUG == 4) ? (shm[0] - 8'd1) : shm[0];
                    v = v >>> sh;
                    if (BUG != 2 && v < 0)    v = 0;
                    if (BUG == 1 && v > 63)   v = 63;
                    if (BUG != 1 && v > 127)  v = 127;
                    h[n] <= v[7:0];
                    acc <= 0; i <= 0;
                    if (n == 15) begin st <= L2; n <= 0; end
                    else begin st <= L1; n <= n + 1; end
                end
                L2: begin                                  // acc += w2[n][i] * h[i]
                    acc <= acc + w2[n*16 + i] * h[i];
                    if (i == 15) st <= L2F; else i <= i + 1;
                end
                L2F: begin                                 // bias + running argmax
                    v = (BUG == 5) ? acc : (acc + b2[n]);
                    if (n == 0 || v > best) begin nb = v; ni = n[2:0]; end
                    else begin nb = best; ni = besti; end
                    best <= nb; besti <= ni;
                    acc <= 0; i <= 0;
                    if (n == 3) begin action <= ni; st <= DN; end
                    else begin n <= n + 1; st <= L2; end
                end
                DN: begin done <= 1; busy <= 0; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
