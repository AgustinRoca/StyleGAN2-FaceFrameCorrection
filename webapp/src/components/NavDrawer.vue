<template>
    <v-navigation-drawer permanent>
      <v-list-item
            prepend-avatar="https://randomuser.me/api/portraits/men/1.jpg"
            title="Admin"
            subtitle="admin@gmail.com"
          ></v-list-item>

      <v-divider></v-divider>

      <v-list
        dense
        nav
      >
        <v-list-item
          v-for="item in items"
          :key="item.title"
          link
          :to="item.to"
        >
          <v-list-item-icon class="me-2">
            <v-icon :icon="item.icon"></v-icon>
          </v-list-item-icon>

          <v-list-item-content>
            <v-list-item-title>{{ item.title }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>
</template>

<script>
import { defineComponent } from 'vue';

export default defineComponent({
    name: 'NavDrawer',
    data () {
      return {
        items: [
          { title: 'Home', icon: 'mdi-home', to: '/' },
          { title: 'Faces', icon: 'mdi-emoticon', to: '/faces' },
          { title: 'Search', icon: 'mdi-magnify', to: '/search' },
          { title: 'Transitions', icon: 'mdi-animation', to: '/transitions' },
          { title: 'From Image', icon: 'mdi-image-move', to: '/fromImage' },
          { title: 'Features', icon: 'fa: fa-solid fa-arrows-to-eye', to: '/features' },
          { title: 'Interchange styles', icon: 'mdi-swap-horizontal', to: '/interchange' },
          { title: 'About', icon: 'mdi-information', to: '/about' },
        ],
        right: null,
      }
    },
})
</script>

<style>

</style>