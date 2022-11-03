<template>
  <v-container style="height:70vh; width: 70vw">
    <v-row justify="center" align="center" class="mt-5">
      <h1>Welcome to Face Generator!</h1>
    </v-row>
    <v-row justify="center" align="center" class="mt-5">
      <h4>Choose what you want to do in the menu on the side</h4>
    </v-row>
    <v-row justify="center" align="center" class="mt-5">
      <v-col md="12" class="d-flex justify-center">
          <v-img contain max-width="50%" src="@/assets/home.jpg" />
      </v-col>
      <v-spacer></v-spacer>
    </v-row>
  </v-container>
</template>

<script>
import { defineComponent } from 'vue';

// Components

export default defineComponent({
  name: 'HomeView',
});
</script>
