cmake_minimum_required(VERSION 3.22)
project(Orbit VERSION 0.4.2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_library(orbit_dsp STATIC Source/OrbitEngine.cpp)
target_include_directories(orbit_dsp PUBLIC Source)
if(MSVC)
    target_compile_options(orbit_dsp PRIVATE /W4 /fp:precise)
else()
    target_compile_options(orbit_dsp PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_executable(orbit_dsp_tests Tests/EngineTests.cpp)
target_link_libraries(orbit_dsp_tests PRIVATE orbit_dsp)
enable_testing()
add_test(NAME orbit_dsp_tests COMMAND orbit_dsp_tests)

option(ORBIT_BUILD_PLUGIN "Build the JUCE VST3 and standalone application" ON)
if(ORBIT_BUILD_PLUGIN)
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.6 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(JUCE)
    juce_add_binary_data(OrbitAssets SOURCES ../web/fonts/oxanium.ttf)
    juce_add_plugin(Orbit
        COMPANY_NAME "Bruce Li"
        BUNDLE_ID "com.bruceyxli.orbit"
        PLUGIN_MANUFACTURER_CODE Brli
        PLUGIN_CODE Orbt
        FORMATS VST3 Standalone
        PRODUCT_NAME "ORBIT"
        IS_SYNTH FALSE NEEDS_MIDI_INPUT FALSE NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        VST3_CATEGORIES Fx Granular
        COPY_PLUGIN_AFTER_BUILD FALSE)
    target_sources(Orbit PRIVATE Source/PluginProcessor.cpp Source/PluginEditor.cpp)
    target_compile_definitions(Orbit PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)
    target_link_libraries(Orbit PRIVATE orbit_dsp OrbitAssets
        juce::juce_audio_utils juce::juce_gui_extra
        PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
    juce_add_console_app(OrbitChecks PRODUCT_NAME "OrbitChecks")
    target_sources(OrbitChecks PRIVATE Tests/HostChecks.cpp)
    target_include_directories(OrbitChecks PRIVATE Source)
    target_link_libraries(OrbitChecks PRIVATE Orbit juce::juce_audio_utils juce::juce_gui_extra)
    add_test(NAME orbit_host_checks COMMAND OrbitChecks)
endif()
