<template>
  <div>
    <h1>Look for faces in DB</h1>
    <SearchForm @generated="facesGenerated"/>
    <SearchModal :showModal="showModal" :imgs_bytes="imgs_bytes" :ids="ids" @close="closeModal"/>
  </div>
</template>
<script>
import { defineComponent } from 'vue';
import SearchForm from '../components/SearchForm.vue'
import SearchModal from '../components/SearchModal.vue'

// Components

export default defineComponent({
  name: 'SearchView',
  components: {
    SearchForm,
    SearchModal,
  },
  data() {
    return {
      showModal: false,
      ids: null,
      imgs_bytes: null
    }
  },
  methods: {
    facesGenerated(imgs_bytes, ids){
      this.showModal = true
      this.ids = ids
      this.imgs_bytes = imgs_bytes
    },
    closeModal(){
      this.showModal = false
    }
  }
});
</script>
