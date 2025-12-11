#include <libhackrf/hackrf.h>
#include <fftw3.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "cxxopts.hpp"

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
static volatile bool should_exit = false;

std::vector<int16_t> rx_i;
std::vector<int16_t> rx_q;
size_t samples_needed = 0;
size_t samples_collected = 0;

// ------------------------------------------------------------
// CTRL-C handler
// ------------------------------------------------------------
void sigint_handler(int)
{
    should_exit = true;
}

// ------------------------------------------------------------
// HackRF RX callback
// ------------------------------------------------------------
int rx_callback(hackrf_transfer* transfer)
{
    size_t count = transfer->valid_length / 2;

    for (size_t i = 0; i < count && samples_collected < samples_needed; i++) {
        rx_i[samples_collected] = (int8_t)transfer->buffer[2*i];
        rx_q[samples_collected] = (int8_t)transfer->buffer[2*i+1];
        samples_collected++;
    }

    if (samples_collected >= samples_needed) {
        should_exit = true;
    }

    return 0;
}

// ------------------------------------------------------------
// Window functions
// ------------------------------------------------------------
std::vector<double> make_window(size_t N, const std::string& type)
{
    std::vector<double> w(N);

    if (type == "none") {
        std::fill(w.begin(), w.end(), 1.0);
    }
    else if (type == "hann") {
        for (size_t i = 0; i < N; i++)
            w[i] = 0.5 * (1 - cos(2*M_PI*i/(N-1)));
    }
    else if (type == "hamming") {
        for (size_t i = 0; i < N; i++)
            w[i] = 0.54 - 0.46*cos(2*M_PI*i/(N-1));
    }
    else if (type == "blackman") {
        for (size_t i = 0; i < N; i++)
            w[i] = 0.42 - 0.5*cos(2*M_PI*i/(N-1)) + 0.08*cos(4*M_PI*i/(N-1));
    }
    else if (type == "blackman-harris") {
        for (size_t i = 0; i < N; i++) {
            w[i] = 0.35875
                 - 0.48829*cos(2*M_PI*i/(N-1))
                 + 0.14128*cos(4*M_PI*i/(N-1))
                 - 0.01168*cos(6*M_PI*i/(N-1));
        }
    }
    else {
        std::cerr << "Unknown window type '" << type << "', using none.\n";
        std::fill(w.begin(), w.end(), 1.0);
    }

    return w;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    signal(SIGINT, sigint_handler);

    // --------------------------------------------------------
    // Argument parser
    // --------------------------------------------------------
    cxxopts::Options options("hackrf_fft",
        "HackRF spectrum snapshot with FFT and windowing");

    options.add_options()
        ("f,freq", "Center frequency (Hz)",
            cxxopts::value<double>()->default_value("100e6"))
        ("r,samp-rate", "Sample rate (Hz)",
            cxxopts::value<double>()->default_value("2e6"))
        ("l,lna", "LNA gain (0-40 dB)",
            cxxopts::value<int>()->default_value("32"))
        ("v,vga", "VGA gain (0-62 dB)",
            cxxopts::value<int>()->default_value("20"))
        ("a,amp", "Enable RF front-end amplifier",
            cxxopts::value<bool>()->default_value("false"))
        ("A,antenna", "Enable antenna power (bias tee)",
            cxxopts::value<bool>()->default_value("false"))
        ("n,fft-size", "FFT size",
            cxxopts::value<int>()->default_value("4096"))
        ("w,window", "Window type (hann, hamming, blackman, "
                     "blackman-harris, none)",
            cxxopts::value<std::string>()->default_value("hann"))
        ("h,help", "Print usage");

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    double  freq_hz     = args["freq"].as<double>();
    double  samp_rate   = args["samp-rate"].as<double>();
    int     lna_gain    = args["lna"].as<int>();
    int     vga_gain    = args["vga"].as<int>();
    bool    amp_enabled = args["amp"].as<bool>();
    bool    ant_power   = args["antenna"].as<bool>();
    int     fft_size    = args["fft-size"].as<int>();
    std::string window_type = args["window"].as<std::string>();

    samples_needed = fft_size;
    rx_i.resize(fft_size);
    rx_q.resize(fft_size);

    // --------------------------------------------------------
    // Print config
    // --------------------------------------------------------
    std::cerr << "Config:\n";
    std::cerr << "  Frequency     : " << freq_hz << " Hz\n";
    std::cerr << "  Sample rate   : " << samp_rate << " Hz\n";
    std::cerr << "  FFT size      : " << fft_size << "\n";

    // --------------------------------------------------------
    // HackRF setup
    // --------------------------------------------------------
    hackrf_init();

    hackrf_device* device = nullptr;
    if (hackrf_open(&device) != HACKRF_SUCCESS) {
        std::cerr << "hackrf_open() failed.\n"; return 1;
    }

    hackrf_set_freq(device, (uint64_t)freq_hz);
    hackrf_set_sample_rate(device, (uint32_t)samp_rate);
    hackrf_set_lna_gain(device, lna_gain);
    hackrf_set_vga_gain(device, vga_gain);
    hackrf_set_amp_enable(device, amp_enabled);
    hackrf_set_antenna_enable(device, ant_power);

    samples_collected = 0;
    hackrf_start_rx(device, rx_callback, nullptr);

    while (!should_exit)
        usleep(1000);

    hackrf_stop_rx(device);
    hackrf_close(device);
    hackrf_exit();

    // --------------------------------------------------------
    // FFT
    // --------------------------------------------------------
    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*fft_size);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*fft_size);

    fftw_plan plan = fftw_plan_dft_1d(
        fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    auto win = make_window(fft_size, window_type);

    for (int i = 0; i < fft_size; i++) {
        in[i][0] = rx_i[i] * win[i];
        in[i][1] = rx_q[i] * win[i];
    }

    fftw_execute(plan);

    // --------------------------------------------------------
    // Output FFT with freq axis in MHz
    // --------------------------------------------------------
    double bin_hz = samp_rate / fft_size;

    std::cout << "Freq_MHz,Power_dB\n";

    for (int k = 0; k < fft_size; k++) {

    // FFT shift
    int k_shifted = (k + fft_size/2) % fft_size;

    // Frequency of this bin (centered at freq_hz)
    double freq_hz_bin = (freq_hz - samp_rate/2.0) + k * bin_hz;
    double freq_mhz = freq_hz_bin / 1e6;

    // Magnitude

    // Normalize FFT amplitude by (N) and ADC range
double re = out[k_shifted][0] / fft_size;
double im = out[k_shifted][1] / fft_size;

// Convert HackRF raw ADC (-128..+127) to ±1.0 scale
re /= 128.0;
im /= 128.0;

double mag = sqrt(re*re + im*im);

// Now convert to dBFS (negative)
double db = 20 * log10(mag + 1e-20);

    std::cout << freq_mhz << "," << db << "\n";
    }
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // --------------------------------------------------------
    // Write gnuplot script
    // --------------------------------------------------------
std::ofstream gp("plot_fft.gnuplot");
gp << "set xlabel 'Frequency (MHz)'\n";
gp << "set ylabel 'Power (dB)'\n";
gp << "set grid\n";
gp << "set datafile separator ','\n";
gp << "plot 'fft.csv' using 1:2 with lines title 'FFT'\n";
gp.close();
    std::cerr << "Wrote gnuplot script: plot_fft.gnuplot\n";

    return 0;
}

