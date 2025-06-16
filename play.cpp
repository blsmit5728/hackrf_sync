#include "libhackrf/hackrf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <fstream>
#include <vector>

#define SAMPLE_RATE_HZ 10000000
#define FREQUENCY_HZ   915000000
#define BUFFER_SIZE    262144  // HackRF default buffer size
#define NUM_BUFFERS    32

std::atomic<bool> running(true);

// Callback for HackRF streaming
int rx_callback(hackrf_transfer* transfer) {
    std::ofstream* outfile = reinterpret_cast<std::ofstream*>(transfer->rx_ctx);
    if (outfile && transfer->buffer) {
        outfile->write(reinterpret_cast<char*>(transfer->buffer), transfer->valid_length);
    }
    return (running.load()) ? 0 : -1;  // Stop when running is false
}

void stream_device(hackrf_device* dev, const char* filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    int status = hackrf_start_rx(dev, rx_callback, &outfile);
    if (status != HACKRF_SUCCESS) {
        std::cerr << "Failed to start RX: " << hackrf_error_name((hackrf_error)status) << std::endl;
        return;
    }

    // Wait for streaming to finish
    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    hackrf_stop_rx(dev);
    outfile.close();
}


int main() {
    if (hackrf_init() != HACKRF_SUCCESS) {
        std::cerr << "Failed to initialize HackRF library." << std::endl;
        return EXIT_FAILURE;
    }
    hackrf_device_list_t * dev_list;
    dev_list = hackrf_device_list();
    if (dev_list->devicecount < 2) {
        std::cerr << "Need at least two HackRF devices. Found: " << dev_list->devicecount << std::endl;
        hackrf_exit();
        return EXIT_FAILURE;
    }

    std::vector< hackrf_device *> devices = {nullptr, nullptr};

    int status = HACKRF_SUCCESS;

    for( int i = 0; i < dev_list->devicecount; i ++)
    {
        std::cout << "Found HackRF SN " << dev_list->serial_numbers[i] << "\n";
        status = hackrf_open_by_serial(dev_list->serial_numbers[i], &devices.at(i));
        if (status != HACKRF_SUCCESS )
        {
            std::cerr << "Error on opening device " << hackrf_error_name((hackrf_error)status) << std::endl;
        }
    }
    
    for( auto it = devices.begin(); it != devices.end(); ++it )
    {
        status = hackrf_set_sample_rate(*it, SAMPLE_RATE_HZ);
        if (status != HACKRF_SUCCESS )
        {
            std::cerr << "Error on opening device " << hackrf_error_name((hackrf_error)status) << std::endl;
        }
        status = hackrf_set_freq(*it, FREQUENCY_HZ);
        if (status != HACKRF_SUCCESS )
        {
            std::cerr << "Error on opening device " << hackrf_error_name((hackrf_error)status) << std::endl;
        }
        status = hackrf_set_lna_gain(*it, 32); // max 40
        if (status != HACKRF_SUCCESS )
        {
            std::cerr << "Error on opening device " << hackrf_error_name((hackrf_error)status) << std::endl;
        }
        status = hackrf_set_vga_gain(*it, 20); // max 62
        if (status != HACKRF_SUCCESS )
        {
            std::cerr << "Error on opening device " << hackrf_error_name((hackrf_error)status) << std::endl;
        }
        read_partid_serialno_t ser;
        hackrf_board_partid_serialno_read(*it, &ser);
        if( ser.serial_no[3] == 0x2f9b4e63)
        {
            hackrf_set_clkout_enable(*it, 1);
        }
    }


    uint8_t clkin_status = 0;
    status = hackrf_get_clkin_status(devices.at(1), &clkin_status);
    printf("CLKIN status: %s\n",
		       clkin_status ? "clock signal detected" :
				      "no clock signal detected");

    // if (hackrf_open_by_serial("00000000000000001a4657dc2f23a53f") == HACKRF_SUCCESS) {
    //     hackrf_open_by_serial("00000000000000001a4657dc2f23a53f", &dev1);
    // }
    // if (hackrf_open_by_serial("00000000000000001a4657dc2f23a540") == HACKRF_SUCCESS) {
    //     hackrf_open_by_serial("00000000000000001a4657dc2f23a540", &dev2);
    // }

    // if (!dev1 || !dev2) {
    //     std::cerr << "Failed to open both HackRF devices." << std::endl;
    //     hackrf_close(dev1);
    //     hackrf_close(dev2);
    //     hackrf_exit();
    //     return EXIT_FAILURE;
    // }

    // // Configure both devices
    // auto setup_device = [](hackrf_device* dev) {
    //     hackrf_set_sample_rate(dev, SAMPLE_RATE_HZ);
    //     hackrf_set_freq(dev, FREQUENCY_HZ);
    //     hackrf_set_lna_gain(dev, 32); // max 40
    //     hackrf_set_vga_gain(dev, 20); // max 62
    // };

    // setup_device(dev1);
    // setup_device(dev2);

    // // Start streaming in separate threads
    std::thread t1(stream_device, devices.at(0), "hackrf1_output.bin");
    std::thread t2(stream_device, devices.at(1), "hackrf2_output.bin");

    std::cout << "Streaming from both HackRFs. Press Enter to stop..." << std::endl;
    std::cin.get();

    running = false;
    t1.join();
    t2.join();
    for( auto it = devices.begin(); it != devices.end(); ++it)
    {
        hackrf_close(*it);
    }
    
    hackrf_exit();

    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
}
