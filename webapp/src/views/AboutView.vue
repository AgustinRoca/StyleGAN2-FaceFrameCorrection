<template>
  <div class="mr-4">
    <h1>About us</h1>
    <h2 class="mt-4">Who made this?</h2>
    <p>This is a final project for Software Engineering in ITBA. It was started by Jimena Lozano 
        and Maite Herrán, and now it is continued by Nicolás Britos and Agustín Roca. From the
        beginning this project was tutored by Rodrigo Ramele.
    </p>
    <h2 class="mt-4">Why?</h2>
    <p><i><a href="https://www.labsuenoymemoria.com">El Laboratorio de Sueño y Memoria</a></i> from 
        Instituto Tecnológico de Buenos Aires (ITBA) studies the formation of false memories, and how 
        these can be reduced or modified, and is in collaboration with the 
        <a href="https://innocenceprojectargentina.org/">Innocence Project</a> to investigate 
        how these can lead to errors in convictions. From this research, the need arises to carry out 
        experiments with human faces that are similar to each other, and how this similarity can result in 
        the formation of false memories.
    </p>
    <p class="mt-2">In this project, we investigate a field of Artificial Intelligence (AI), Deep Learning, which
        can provide us with a solution to the generation of artificial faces. In particular, we
        implement a face generation model using a Generative Adversarial Network (GAN), with
        the aim of generating faces as realistic as possible, so that a human cannot distinguish
        them from real faces. StyleGAN, a particular implementation of the GAN network, was
        the chosen architecture, because in addition to producing images with high resolution
        quality, it presents a model that allows navigation of the latent space and the synthesis of
        faces, using style mixing properties.
    </p>
  </div>
</template>

<script>
import { defineComponent } from 'vue';

// Components

export default defineComponent({
  name: 'AboutView',
});
</script>