<template>
  <div id="app">
    <h1>Face Verification System</h1>
    <WebcamCapture />
  </div>
</template>

<script setup>
import WebcamCapture from './components/WebcamCapture.vue'
</script>

<style>
#app {
  text-align: center;
  margin-top: 20px;
}
</style>